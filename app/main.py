# FILE: app/main.py
from __future__ import annotations

from .fast import speed_tweaks
speed_tweaks()

import asyncio
import json
import re
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

app = FastAPI(title="Basaa Realtime Pipeline", version="1.0")

# ─────────────────────────── models ───────────────────────────
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

# ─────────────────────────── helpers ───────────────────────────
_BASAA_CHARS = set("ŋɓƁàáâèéêìíîòóôùúûɛɔ")
_FR_DIACRIT  = set("àâäæçéèêëîïôœùûüÿ")
_FR_COMMON   = {" le ", " la ", " les ", " des ", " de ", " du ", " un ", " une ",
                " je ", " tu ", " il ", " elle ", " nous ", " vous ", " et ",
                " que ", " qui ", " est ", " pour ", " avec ", " pas "}

def _guess_lang_code(s: str) -> str:
    """
    Very light heuristic to route text → M2M safely.
    Returns one of {"lg","fr","en"}.
    """
    if not s:
        return "fr"
    l = f" {s.lower()} "
    if any(c in _BASAA_CHARS for c in l):
        return "lg"
    if any(c in _FR_DIACRIT for c in l) or any(w in l for w in _FR_COMMON):
        return "fr"
    letters = re.sub(r"[^a-z]", "", l)
    return "en" if len(letters) >= 1 else "fr"

async def _safe_close(ws: WebSocket):
    try:
        if getattr(ws, "client_state", None) != WebSocketState.DISCONNECTED:
            await ws.close()
    except Exception:
        pass

async def _pcm_sender(ws: WebSocket, q: "asyncio.Queue[bytes]"):
    """
    Single-task sender that serializes ws.send_bytes. Put b"...pcm..." into q,
    and finally put None as a sentinel to stop the task.
    """
    try:
        while True:
            data = await q.get()
            if data is None:
                break
            if getattr(ws, "application_state", None) == WebSocketState.DISCONNECTED:
                break
            if getattr(ws, "client_state", None) == WebSocketState.DISCONNECTED:
                break
            try:
                await ws.send_bytes(data)
            except Exception:
                break
    finally:
        try:
            while not q.empty():
                q.get_nowait()
        except Exception:
            pass

# ─────────────────────────── health ───────────────────────────
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# ─────────────────────────── TEXT endpoints ───────────────────────────
# NOTE: These return **Basaa only** (raw text) — no JSON echo.
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        src = _guess_lang_code(text)
        out_lg = text if src == "lg" else MT.to_lg(text, src)
        await ws.send_text(out_lg)  # Basaa only
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

@app.websocket("/ws/chat_text")
async def ws_chat_text(ws: WebSocket):
    await ws.accept()
    try:
        user_text = (await ws.receive_text()).strip()
        src = _guess_lang_code(user_text)
        user_fr = user_text if src == "fr" else MT.to_fr(user_text, src)
        qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")
        await ws.send_text(basaa)     # Basaa only
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# ─────────────────────────── non-streaming AUDIO ───────────────────────────
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
    wav = TTS.tts(lg_text)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    await ws.send_text(json.dumps({"asr": {"text": text, "lang": wlang},
                                   "fr": fr_text, "lg": lg_text}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
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
    wav     = TTS.tts(basaa)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    await ws.send_text(json.dumps({"asr": {"text": user_text, "lang": wlang},
                                   "fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
    await _safe_close(ws)

# ─────────────────────────── streaming AUDIO ───────────────────────────
@app.websocket("/ws/translate_stream")
async def ws_translate_stream(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"] == "DONE":
                    break
                if "bytes" in msg and msg["bytes"]:
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

    try:
        await ws.send_text(json.dumps({"asr": {"text": text, "lang": wlang},
                                       "fr": fr_text, "lg": lg_text}, ensure_ascii=False))
        await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))
    except Exception:
        await _safe_close(ws); return

    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
    sender_task = asyncio.create_task(_pcm_sender(ws, q))

    try:
        def on_chunk(pcm: np.ndarray):
            if sender_task.done():
                return
            try:
                q.put_nowait(pcm.astype(np.float32, copy=False).tobytes(order="C"))
            except asyncio.QueueFull:
                pass

        TTS.stream_tts(lg_text, on_chunk=on_chunk, chunk_ms=600)
    except Exception:
        traceback.print_exc()
    finally:
        try: q.put_nowait(None)
        except Exception: pass
        try: await asyncio.wait_for(sender_task, timeout=3.0)
        except Exception: pass
        try: await ws.send_text("DONE")
        except Exception: pass
        await _safe_close(ws)

@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"] == "DONE":
                    break
                if "bytes" in msg and msg["bytes"]:
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

    try:
        await ws.send_text(json.dumps({"asr": {"text": user_text, "lang": wlang},
                                       "fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
        await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))
    except Exception:
        await _safe_close(ws); return

    q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
    sender_task = asyncio.create_task(_pcm_sender(ws, q))

    try:
        def on_chunk(pcm: np.ndarray):
            if sender_task.done():
                return
            try:
                q.put_nowait(pcm.astype(np.float32, copy=False).tobytes(order="C"))
            except asyncio.QueueFull:
                pass

        TTS.stream_tts(basaa, on_chunk=on_chunk, chunk_ms=600)
    except Exception:
        traceback.print_exc()
    finally:
        try: q.put_nowait(None)
        except Exception: pass
        try: await asyncio.wait_for(sender_task, timeout=3.0)
        except Exception: pass
        try: await ws.send_text("DONE")
        except Exception: pass
        await _safe_close(ws)
