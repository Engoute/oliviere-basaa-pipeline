# FILE: app/main.py
from __future__ import annotations

from .fast import speed_tweaks
speed_tweaks()

import asyncio
import json
import re
import unicodedata
import traceback
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M
from .tts_orpheus import Orpheus
from .llm_qwen import QwenAgent
from .utils_audio import wav_bytes_from_float32

app = FastAPI(title="Basaa Realtime Pipeline", version="0.92")

# ---- model singletons --------------------------------------------------------
print(f"[main] Initializing models…")
print(f"[main]  PATH_WHISPER = {S.path_whisper}")
ASR_MODEL = ASR(S.path_whisper)

print(f"[main]  PATH_M2M     = {S.path_m2m}")
MT        = M2M(S.path_m2m)

print(f"[main]  PATH_ORPHEUS  = {S.path_orpheus}")
TTS       = Orpheus(S.path_orpheus, sr_out=S.tts_sr)

print(f"[main]  PATH_QWEN     = {S.path_qwen}")
QWEN      = QwenAgent(S.path_qwen)

print(f"[main] Models ready.")

# ---- small helpers -----------------------------------------------------------
async def _safe_close(ws: WebSocket):
    try:
        if getattr(ws, "client_state", None) != WebSocketState.DISCONNECTED:
            await ws.close()
    except Exception:
        pass

def _looks_basaa(text: str) -> bool:
    t = text.lower()
    return any(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in t)

def _normalize_tts_text(text: str) -> str:
    # keep letters, marks, digits, common punct & space; collapse whitespace; ensure trailing period
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("L", "M", "N"):
            kept.append(ch); continue
        if ch in " .,!?:;'\"-()[]/\\\n\r\t":
            kept.append(ch)
    s = "".join(kept)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?": s += "."
    return s

def _split_for_tts(text: str, max_len: int = 240) -> list[str]:
    # sentence-ish split, then soft-wrap to max_len
    if not text: return []
    pieces = re.split(r"(?<=[\.\!\?])\s+", text)
    out: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p: continue
        if len(p) <= max_len:
            out.append(p); continue
        # soft wrap on whitespace
        cur: list[str] = []
        for tok in re.split(r"(\s+)", p):
            if sum(len(x) for x in cur) + len(tok) > max_len and cur:
                out.append("".join(cur).strip())
                cur = [tok]
            else:
                cur.append(tok)
        if cur:
            out.append("".join(cur).strip())
    return out

def _is_bad_wave(w: np.ndarray) -> bool:
    if w is None or not isinstance(w, np.ndarray) or w.size == 0:
        return True
    if not np.all(np.isfinite(w)):
        return True
    # absurdly short (e.g., stuck)
    if w.shape[0] < int(0.10 * S.tts_sr):  # <100ms
        return True
    # almost-DC or tiny variance → likely bad
    if float(np.std(w)) < 1e-5:
        return True
    return False

def synthesize_wav_safe(text: str) -> np.ndarray:
    """
    Non-streaming, robust synthesis:
      - normalize & sentence-split
      - synth each sentence; retry once with simplified text if needed
      - concat; ensure not crazy loud
      - fall back to short silence on failure
    """
    clean = _normalize_tts_text(text)
    chunks = _split_for_tts(clean)
    if not chunks:
        return np.zeros(int(0.3 * S.tts_sr), dtype=np.float32)

    waves: list[np.ndarray] = []
    for chunk in chunks:
        wav = None
        try:
            wav = TTS.tts(chunk)  # float32 mono @ S.tts_sr
        except Exception:
            wav = None

        if _is_bad_wave(wav):
            # retry with simplified text (strip non-word/punct, collapse spaces)
            simple = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", "", chunk)
            simple = re.sub(r"\s+", " ", simple).strip()
            try:
                wav = TTS.tts(simple)
            except Exception:
                wav = None

        if _is_bad_wave(wav):
            # fallback: ~200ms silence for this chunk
            wav = np.zeros(int(0.2 * S.tts_sr), dtype=np.float32)

        waves.append(wav.astype(np.float32, copy=False))

    out = np.concatenate(waves, dtype=np.float32) if waves else np.zeros(0, dtype=np.float32)
    # light limiter
    if out.size:
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:
            out = out * (0.99 / peak)
    return out

# ------------------------------------------------------------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------- text endpoints ------------------------------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        is_basaa = _looks_basaa(text)
        out_fr = MT.to_fr(text, "lg") if is_basaa else text
        out_lg = text if is_basaa else MT.to_lg(text, "fr")
        # Basaa-only is what the app shows; JSON keeps future-flexibility.
        await ws.send_text(json.dumps({"fr": out_fr, "lg": out_lg}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

@app.websocket("/ws/chat_text")
async def ws_chat_text(ws: WebSocket):
    await ws.accept()
    try:
        user_text = (await ws.receive_text()).strip()
        qwen_fr = QWEN.chat_fr(user_text, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")
        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- TTS once (non-streaming WAV) ----------------------------
@app.websocket("/ws/tts_once")
async def ws_tts_once(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        wav = synthesize_wav_safe(text)
        wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)
        try:
            await ws.send_bytes(wav_bytes)
        except Exception:
            # client might have closed quickly; ignore
            pass
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- legacy non-streaming audio ------------------------------
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    if msg["text"] == "DONE":
                        break
                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception:
        traceback.print_exc(); await _safe_close(ws); return

    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    if wlang == "lg":
        fr_text, lg_text = MT.to_fr(text, "lg"), text
    else:
        fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
        lg_text = MT.to_lg(text, wlang)

    # send text json first
    try:
        await ws.send_text(json.dumps({"asr": {"text": text, "lang": wlang},
                                       "fr": fr_text, "lg": lg_text}, ensure_ascii=False))
    except Exception:
        await _safe_close(ws); return

    # synth full wav once
    try:
        wav = synthesize_wav_safe(lg_text)
        wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)
        try:
            await ws.send_bytes(wav_bytes)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

@app.websocket("/ws/audio_chat")
async def ws_audio_chat(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    if msg["text"] == "DONE":
                        break
                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception:
        traceback.print_exc(); await _safe_close(ws); return

    user_text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    user_fr = user_text if wlang == "fr" else MT.to_fr(user_text, wlang)
    qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
    basaa   = MT.to_lg(qwen_fr, "fr")

    # send text first
    try:
        await ws.send_text(json.dumps({"asr": {"text": user_text, "lang": wlang},
                                       "fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    except Exception:
        await _safe_close(ws); return

    # synth full wav once
    try:
        wav = synthesize_wav_safe(basaa)
        wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)
        try:
            await ws.send_bytes(wav_bytes)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- "streaming" endpoints now send a full WAV ----------------
@app.websocket("/ws/translate_stream")
async def ws_translate_stream(ws: WebSocket):
    # identical to /ws/translate, kept for client compatibility
    await ws_translate(ws)

@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(ws: WebSocket):
    # identical to /ws/audio_chat, kept for client compatibility
    await ws_audio_chat(ws)
 
