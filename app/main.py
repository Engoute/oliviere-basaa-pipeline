# FILE: app/main.py
from __future__ import annotations

from .fast import speed_tweaks
speed_tweaks()

import asyncio
import json
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

app = FastAPI(title="Basaa Realtime Pipeline", version="0.9")

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

# ---- helpers -----------------------------------------------------------------
async def _safe_close(ws: WebSocket):
    try:
        if getattr(ws, "client_state", None) != WebSocketState.DISCONNECTED:
            await ws.close()
    except Exception:
        pass

async def _ws_sender(ws: WebSocket, q: "asyncio.Queue[tuple[str, bytes | str]]"):
    """
    Serialize send() calls. Queue items are ('bin', bytes) or ('txt', str).
    End with ('end', b'').
    """
    try:
        while True:
            kind, data = await q.get()
            if kind == "end":
                break
            if getattr(ws, "application_state", None) == WebSocketState.DISCONNECTED:
                break
            if getattr(ws, "client_state", None) == WebSocketState.DISCONNECTED:
                break
            try:
                if kind == "bin":
                    await ws.send_bytes(data)         # type: ignore[arg-type]
                else:
                    await ws.send_text(data)          # type: ignore[arg-type]
            except Exception:
                break
    finally:
        # drain
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass

# ------------------------------------------------------------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------- TEXT endpoints (Basaa-only payloads) --------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        is_basaa = any(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in text.lower())
        # Basaa-only output for text mode
        out_lg = text if is_basaa else MT.to_lg(text, "fr")
        await ws.send_text(json.dumps({"lg": out_lg}, ensure_ascii=False))
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
        # Basaa-only output for text mode
        await ws.send_text(json.dumps({"lg": basaa}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- Non-streaming TTS (for long-press “Lire à voix haute”) --
@app.websocket("/ws/tts_once")
async def ws_tts_once(ws: WebSocket):
    await ws.accept()
    try:
        txt = (await ws.receive_text()).strip()
        wav = TTS.tts(txt)
        if wav.size > 0:
            await ws.send_bytes(wav_bytes_from_float32(wav, S.tts_sr))
        else:
            await ws.send_text(json.dumps({"error": "tts_empty"}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"error": "tts_exception"}, ensure_ascii=False))
        except Exception:
            pass
    finally:
        await _safe_close(ws)

# -------------------- legacy non-streaming AUDIO ------------------------------
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    if msg["text"] == "DONE": break
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
                    if msg["text"] == "DONE": break
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

# -------------------- STREAMING AUDIO (header-after-first-chunk) --------------
def _pack_pcm(pcm: np.ndarray) -> bytes:
    return pcm.astype(np.float32, copy=False).tobytes(order="C")

async def _stream_or_fallback(ws: WebSocket, text_for_tts: str):
    """
    Stream PCM if we get chunks; otherwise send a single WAV fallback.
    Header is sent *after* the first chunk so clients don't hang if the
    model yields nothing.
    """
    q: asyncio.Queue[tuple[str, bytes | str]] = asyncio.Queue(maxsize=32)
    sender_task = asyncio.create_task(_ws_sender(ws, q))

    chunk_count = 0
    try:
        def on_chunk(pcm: np.ndarray):
            nonlocal chunk_count
            chunk = _pack_pcm(pcm)
            if chunk_count == 0:
                # first chunk → send header as a TEXT frame via the same sender
                try:
                    q.put_nowait(("txt", json.dumps({"sr": S.tts_sr, "format": "pcm_f32"})))
                except asyncio.QueueFull:
                    pass
            chunk_count += 1
            try:
                q.put_nowait(("bin", chunk))
            except asyncio.QueueFull:
                # drop if client can't keep up
                pass

        # This is synchronous; it will invoke on_chunk many times or none.
        TTS.stream_tts(text_for_tts, on_chunk=on_chunk, chunk_ms=600)

    except Exception:
        traceback.print_exc()

    # wrap up
    if chunk_count == 0:
        # No chunks produced → non-streaming fallback
        try:
            wav = TTS.tts(text_for_tts)
            if wav.size > 0:
                await ws.send_bytes(wav_bytes_from_float32(wav, S.tts_sr))
            else:
                await ws.send_text(json.dumps({"error": "tts_empty"}, ensure_ascii=False))
        except Exception:
            traceback.print_exc()
            try:
                await ws.send_text(json.dumps({"error": "tts_exception"}, ensure_ascii=False))
            except Exception:
                pass
        finally:
            try: await ws.send_text("DONE")
            except Exception: pass
            try:
                q.put_nowait(("end", b""))
            except Exception:
                pass
            try:
                await asyncio.wait_for(sender_task, timeout=2.0)
            except Exception:
                pass
            return

    # Had chunks → finish streaming path
    try:
        q.put_nowait(("end", b""))
    except Exception:
        pass
    try:
        await asyncio.wait_for(sender_task, timeout=3.0)
    except Exception:
        pass
    try:
        await ws.send_text("DONE")
    except Exception:
        pass

@app.websocket("/ws/translate_stream")
async def ws_translate_stream(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"] == "DONE": break
                if "bytes" in msg and msg["bytes"]: buf.extend(msg["bytes"])
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

    # send texts first (ASR + translations)
    try:
        await ws.send_text(json.dumps({"asr": {"text": text, "lang": wlang},
                                       "fr": fr_text, "lg": lg_text}, ensure_ascii=False))
    except Exception:
        await _safe_close(ws); return

    await _stream_or_fallback(ws, lg_text)
    await _safe_close(ws)

@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"] == "DONE": break
                if "bytes" in msg and msg["bytes"]: buf.extend(msg["bytes"])
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
    except Exception:
        await _safe_close(ws); return

    await _stream_or_fallback(ws, basaa)
    await _safe_close(ws)
