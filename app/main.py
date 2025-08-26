# FILE: app/main.py
from __future__ import annotations

from .fast import speed_tweaks
speed_tweaks()

import json
import re
import unicodedata
import traceback
from typing import Optional, Tuple

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

app = FastAPI(title="Basaa Realtime Pipeline", version="0.95")

# ---- model singletons --------------------------------------------------------
print(f"[main] Initializing models…")

print(f"[main]  PATH_WHISPER_BASAA    = {S.path_whisper_basaa or S.path_whisper}")
ASR_BASAA   = ASR(S.path_whisper_basaa or S.path_whisper)

print(f"[main]  PATH_WHISPER_GENERAL  = {S.path_whisper_general}")
ASR_GENERAL = ASR(S.path_whisper_general)

print(f"[main]  PATH_M2M              = {S.path_m2m}")
MT   = M2M(S.path_m2m)

print(f"[main]  PATH_ORPHEUS          = {S.path_orpheus}")
TTS  = Orpheus(S.path_orpheus, sr_out=S.tts_sr)

print(f"[main]  PATH_QWEN             = {S.path_qwen}")
QWEN = QwenAgent(S.path_qwen)

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
    if not text: return []
    pieces = re.split(r"(?<=[\.\!\?])\s+", text)
    out: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p: continue
        if len(p) <= max_len:
            out.append(p); continue
        cur: list[str] = []
        for tok in re.split(r"(\s+)", p):
            if sum(len(x) for x in cur) + len(tok) > max_len and cur:
                out.append("".join(cur).strip()); cur = [tok]
            else:
                cur.append(tok)
        if cur: out.append("".join(cur).strip())
    return out

def _is_bad_wave(w: np.ndarray) -> bool:
    if w is None or not isinstance(w, np.ndarray) or w.size == 0: return True
    if not np.all(np.isfinite(w)): return True
    if w.shape[0] < int(0.10 * S.tts_sr): return True
    if float(np.std(w)) < 1e-5: return True
    return False

def synthesize_wav_safe(text: str) -> np.ndarray:
    clean = _normalize_tts_text(text)
    chunks = _split_for_tts(clean)
    if not chunks:
        return np.zeros(int(0.3 * S.tts_sr), dtype=np.float32)

    waves: list[np.ndarray] = []
    for chunk in chunks:
        wav = None
        try:
            wav = TTS.tts(chunk)
        except Exception:
            wav = None

        if _is_bad_wave(wav):
            simple = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", "", chunk)
            simple = re.sub(r"\s+", " ", simple).strip()
            try:
                wav = TTS.tts(simple)
            except Exception:
                wav = None

        if _is_bad_wave(wav):
            wav = np.zeros(int(0.2 * S.tts_sr), dtype=np.float32)

        waves.append(wav.astype(np.float32, copy=False))

    out = np.concatenate(waves, dtype=np.float32) if waves else np.zeros(0, dtype=np.float32)
    if out.size:
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:
            out = out * (0.99 / peak)
    return out

# ---- JSON helpers ------------------------------------------------------------
def _try_parse_json(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s or s[0] not in "{[" or s[-1] not in "}]":
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _norm_lang(c: Optional[str]) -> Optional[str]:
    if not c: return None
    c = c.strip().lower()
    if c in ("fr", "en", "lg"): return c
    if c in ("fra", "fr-fr"):   return "fr"
    if c in ("eng", "en-us", "en-gb"): return "en"
    if c in ("bas", "basaa"):   return "lg"
    return None

# ------------------------------------------------------------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------- text endpoints (accept JSON, return BOTH) ---------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        raw = (await ws.receive_text()).strip()
        obj = _try_parse_json(raw)
        if obj:
            text = (obj.get("text") or obj.get("t") or "").strip()
            src_hint = _norm_lang(obj.get("src") or obj.get("lang"))
        else:
            text = raw
            src_hint = None

        src = MT.resolve_safe_src(text, src_hint)
        out_fr = MT.to_fr(text, src)
        out_lg = MT.to_lg(text, src)
        await ws.send_text(json.dumps({"fr": out_fr, "lg": out_lg}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

@app.websocket("/ws/chat_text")
async def ws_chat_text(ws: WebSocket):
    await ws.accept()
    try:
        raw = (await ws.receive_text()).strip()
        obj = _try_parse_json(raw)
        if obj:
            user_text = (obj.get("text") or obj.get("t") or "").strip()
            lang = _norm_lang(obj.get("lang") or obj.get("src")) or ("lg" if _looks_basaa(user_text) else "fr")
        else:
            user_text = raw
            lang = "lg" if _looks_basaa(user_text) else "fr"

        # Normalize to FR for Qwen
        if lang == "lg":
            user_fr = MT.to_fr(user_text, "lg")
        elif lang == "en":
            user_fr = MT.to_fr(user_text, "en")
        else:
            user_fr = user_text

        qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")
        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- audio helpers (header + bytes) --------------------------
async def _recv_audio_with_lang_header(ws: WebSocket) -> Tuple[bytes, Optional[str]]:
    """
    First text frame may be JSON: {"lang":"fr|en|lg"} or {"src": "..."}.
    Then binary chunks. Finish when client sends "DONE".
    """
    buf = bytearray()
    lang_hint: Optional[str] = None
    first = True
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    s = (msg.get("text") or "").strip()
                    if s == "DONE":
                        break
                    if first:
                        obj = _try_parse_json(s)
                        if obj:
                            lang_hint = _norm_lang(obj.get("lang") or obj.get("src"))
                            first = False
                            continue
                    # otherwise ignore stray text frames
                elif "bytes" in msg and msg["bytes"]:
                    first = False
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
    return bytes(buf), lang_hint

def _choose_asr(lang_hint: Optional[str]) -> ASR:
    # If user explicitly chooses Basaa, force Basaa FT; else use General v3.
    return ASR_BASAA if lang_hint == "lg" else ASR_GENERAL

# -------------------- legacy non-streaming audio ------------------------------
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    try:
        pcm16, lang_hint = await _recv_audio_with_lang_header(ws)
        asr = _choose_asr(lang_hint)
        text, wlang, _ = asr.transcribe(pcm16)

        if wlang == "lg":
            fr_text, lg_text = MT.to_fr(text, "lg"), text
        else:
            fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
            lg_text = MT.to_lg(text, wlang)

        await ws.send_text(json.dumps({"asr": {"text": text, "lang": wlang},
                                       "fr": fr_text, "lg": lg_text}, ensure_ascii=False))

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
    try:
        pcm16, lang_hint = await _recv_audio_with_lang_header(ws)
        asr = _choose_asr(lang_hint)
        user_text, wlang, _ = asr.transcribe(pcm16)

        if wlang == "lg":
            user_fr = MT.to_fr(user_text, "lg")
        elif wlang == "en":
            user_fr = MT.to_fr(user_text, "en")
        else:
            user_fr = user_text if wlang == "fr" else MT.to_fr(user_text, wlang)

        qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")

        await ws.send_text(json.dumps({"asr": {"text": user_text, "lang": wlang},
                                       "fr": qwen_fr, "lg": basaa}, ensure_ascii=False))

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

# -------------------- streaming aliases (compat) ------------------------------
@app.websocket("/ws/translate_stream")
async def ws_translate_stream(ws: WebSocket):
    await ws_translate(ws)

@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(ws: WebSocket):
    await ws_audio_chat(ws)
