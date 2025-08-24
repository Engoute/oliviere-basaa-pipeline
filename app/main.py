# FILE: app/main.py
from __future__ import annotations

from .fast import speed_tweaks
speed_tweaks()

import json, traceback
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M
from .tts_orpheus import Orpheus
from .llm_qwen import QwenAgent
from .utils_audio import wav_bytes_from_float32

app = FastAPI(title="Basaa Realtime Pipeline", version="0.7")

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

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------- text endpoints --------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        # raw text payload
        text = (await ws.receive_text()).strip()
        is_basaa = any(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in text.lower())
        out_fr = MT.to_fr(text, "lg") if is_basaa else text
        out_lg = text if is_basaa else MT.to_lg(text, "fr")
        await ws.send_text(json.dumps({"fr": out_fr, "lg": out_lg}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

@app.websocket("/ws/chat_text")
async def ws_chat_text(ws: WebSocket):
    await ws.accept()
    try:
        user_text = (await ws.receive_text()).strip()
        # Normalize to EN for Qwen prompt semantics if desired – we keep FR here
        qwen_fr = QWEN.chat_fr(user_text, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")
        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- legacy non-streaming audio --------------------
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
        traceback.print_exc(); await ws.close(); return

    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    if wlang == "lg":
        fr_text, lg_text = MT.to_fr(text, "lg"), text
    else:
        fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
        lg_text = MT.to_lg(text, wlang)
    wav = TTS.tts(lg_text)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    await ws.send_text(json.dumps({"asr":{"text":text,"lang":wlang},"fr":fr_text,"lg":lg_text}, ensure_ascii=False))
    if wav_bytes: await ws.send_bytes(wav_bytes)
    await ws.close()

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
        traceback.print_exc(); await ws.close(); return

    user_text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    user_fr = user_text if wlang == "fr" else MT.to_fr(user_text, wlang)
    qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
    basaa   = MT.to_lg(qwen_fr, "fr")
    wav     = TTS.tts(basaa)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    await ws.send_text(json.dumps({"asr":{"text":user_text,"lang":wlang},"fr":qwen_fr,"lg":basaa}, ensure_ascii=False))
    if wav_bytes: await ws.send_bytes(wav_bytes)
    await ws.close()

# -------------------- streaming endpoints --------------------
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
        traceback.print_exc(); await ws.close(); return

    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    if wlang == "lg":
        fr_text, lg_text = MT.to_fr(text, "lg"), text
    else:
        fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
        lg_text = MT.to_lg(text, wlang)

    # Send transcript + translations first (now includes ASR)
    await ws.send_text(json.dumps({"asr":{"text":text,"lang":wlang},"fr":fr_text,"lg":lg_text}, ensure_ascii=False))
    # Audio header
    await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))

    try:
        def on_chunk(pcm: np.ndarray):
            import asyncio
            asyncio.create_task(ws.send_bytes(pcm.tobytes(order="C")))
        TTS.stream_tts(lg_text, on_chunk=on_chunk, chunk_ms=600)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.send_text("DONE")
        await ws.close()

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
        traceback.print_exc(); await ws.close(); return

    user_text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    user_fr = user_text if wlang == "fr" else MT.to_fr(user_text, wlang)
    qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
    basaa   = MT.to_lg(qwen_fr, "fr")

    # Send reply text (with ASR) first
    await ws.send_text(json.dumps({"asr":{"text":user_text,"lang":wlang},"fr":qwen_fr,"lg":basaa}, ensure_ascii=False))
    # Audio header
    await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))

    try:
        def on_chunk(pcm: np.ndarray):
            import asyncio
            asyncio.create_task(ws.send_bytes(pcm.tobytes(order="C")))
        TTS.stream_tts(basaa, on_chunk=on_chunk, chunk_ms=600)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.send_text("DONE")
        await ws.close()
