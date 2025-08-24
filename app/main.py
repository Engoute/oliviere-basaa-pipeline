# FILE: app/main.py
from .fast import speed_tweaks
speed_tweaks()

import json, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M
from .tts_orpheus import Orpheus
from .llm_qwen import QwenAgent
from .utils_audio import wav_bytes_from_float32

app = FastAPI(title="Basaa Realtime Pipeline", version="0.5")

ASR_MODEL = ASR(S.path_whisper)
MT        = M2M(S.path_m2m)
TTS       = Orpheus(S.path_orpheus, sr_out=S.tts_sr)
QWEN      = QwenAgent(S.path_qwen)

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------- existing non-streaming endpoints --------------------
@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
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

    text, lang, conf = ASR_MODEL.transcribe(bytes(buf))
    await ws.send_text(json.dumps({"text": text, "lang": lang, "lang_conf": conf}, ensure_ascii=False))
    await ws.close()

@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        is_basaa = any(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in text.lower())
        out = MT.to_fr(text, "lg") if is_basaa else MT.to_lg(text, "fr")
        await ws.send_text(out)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

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

    wav_bytes = None
    if wlang == "lg":
        fr_text = MT.to_fr(text, "lg")
        lg_text = text
    else:
        fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
        lg_text = MT.to_lg(text, wlang)
        wav = TTS.tts(lg_text)
        if wav.size > 0:
            wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)

    await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
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

    # 1) ASR
    user_text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))

    # 2) Normalize to French
    user_fr = user_text if wlang == "fr" else MT.to_fr(user_text, wlang)

    # 3) Qwen chat (FR persona)
    qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)

    # 4) FR -> Basaa + TTS
    basaa = MT.to_lg(qwen_fr, "fr")
    wav = TTS.tts(basaa)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    # 5) Return JSON + single WAV frame
    await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
    await ws.close()

# -------------------- NEW: streaming endpoints (PCM chunks) --------------------
# Protocol:
#   • Server first sends a JSON header: {"sr": 24000, "format": "pcm_f32"}
#   • Then sends multiple binary frames (float32 mono PCM).
#   • Finally sends a "DONE" text frame and closes.
#
# NOTE: Your current MAUI client buffers a single WAV. To hear audio live, the client
# must stream-play float32 PCM (e.g., Android AudioTrack). Server support is here; client change is next step.

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
        traceback.print_exc(); await ws.close(); return

    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    if wlang == "lg":
        fr_text, lg_text = MT.to_fr(text, "lg"), text
    else:
        fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
        lg_text = MT.to_lg(text, wlang)

    # Send transcript/translation first
    await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))

    # Stream header for audio (PCM float32)
    await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))

    # Stream PCM chunks
    try:
        def on_chunk(pcm: np.ndarray):
            # Send as bytes (float32 little-endian)
            ws_bytes = pcm.tobytes(order="C")
            # Note: awaiting inside callback is tricky; use create_task
            import asyncio
            asyncio.create_task(ws.send_bytes(ws_bytes))

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
                if "text" in msg and msg["text"] == "DONE":
                    break
                if "bytes" in msg and msg["bytes"]:
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
    basaa  = MT.to_lg(qwen_fr, "fr")

    # Send reply text first
    await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))

    # Send audio header
    await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))

    # Stream PCM chunks
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
