# FILE: app/main.py
from __future__ import annotations

from .fast import speed_tweaks
speed_tweaks()

import json, traceback, asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M, looks_like_fr
from .tts_orpheus import Orpheus
from .llm_qwen import QwenAgent
from .utils_audio import wav_bytes_from_float32

app = FastAPI(title="Basaa Realtime Pipeline", version="0.7")

# ---- model singletons ----
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

# -------------------- helpers --------------------
async def _recv_all_binary(ws: WebSocket) -> bytes:
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
                return bytes(buf)
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
    return bytes(buf)

def _ensure_french(s: str) -> str:
    # Qwen *should* answer in FR. If we detect it's not FR, translate EN->FR as a safety net.
    # (We only implement EN->FR fallback because Qwen input is EN in chat mode.)
    if looks_like_fr(s):
        return s
    try:
        return MT.to_fr(s, "en")
    except Exception:
        return s

# -------------------- ASR (non-streaming) --------------------
@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    try:
        data = await _recv_all_binary(ws)
        text, lang, conf = ASR_MODEL.transcribe(data)
        await ws.send_text(json.dumps({"text": text, "lang": lang, "lang_conf": conf}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- TRANSLATE (TEXT) --------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        # RAW text frame from client
        text = (await ws.receive_text()).strip()
        # We do not have Whisper hint here (text input), pass None
        src = MT.resolve_safe_src(text, whisper_hint=None)
        fr_text = MT.to_fr(text, src)
        lg_text = MT.to_lg(text, src)
        await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- CHAT (TEXT) — NEW --------------------
@app.websocket("/ws/chat_text")
async def ws_chat_text(ws: WebSocket):
    await ws.accept()
    try:
        user_text = (await ws.receive_text()).strip()
        src = MT.resolve_safe_src(user_text, whisper_hint=None)

        # Chat pipeline: normalize to EN → Qwen → ensure FR → FR→LG
        user_en = MT.to_en(user_text, src)
        qwen_fr = QWEN.chat_fr(user_en, temperature=0.2)
        qwen_fr = _ensure_french(qwen_fr)
        basaa   = MT.to_lg(qwen_fr, "fr")

        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- TRANSLATE (AUDIO) --------------------
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    try:
        data = await _recv_all_binary(ws)
        text, whisper_hint, _ = ASR_MODEL.transcribe(data)
        src = MT.resolve_safe_src(text, whisper_hint)

        fr_text = MT.to_fr(text, src)
        lg_text = MT.to_lg(text, src)

        wav_bytes = None
        # Always TTS Basaa in Translate mode
        wav = TTS.tts(lg_text)
        if wav.size > 0:
            wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)

        await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))
        if wav_bytes:
            await ws.send_bytes(wav_bytes)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- CHAT (AUDIO) --------------------
@app.websocket("/ws/audio_chat")
async def ws_audio_chat(ws: WebSocket):
    await ws.accept()
    try:
        data = await _recv_all_binary(ws)
        user_text, whisper_hint, _ = ASR_MODEL.transcribe(data)
        src = MT.resolve_safe_src(user_text, whisper_hint)

        # Chat pipeline: to EN → Qwen (FR) → ensure FR → FR→LG → TTS(LG)
        user_en = MT.to_en(user_text, src)
        qwen_fr = QWEN.chat_fr(user_en, temperature=0.2)
        qwen_fr = _ensure_french(qwen_fr)
        basaa   = MT.to_lg(qwen_fr, "fr")

        wav_bytes = None
        wav = TTS.tts(basaa)
        if wav.size > 0:
            wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)

        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
        if wav_bytes:
            await ws.send_bytes(wav_bytes)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- TRANSLATE (AUDIO STREAM) --------------------
@app.websocket("/ws/translate_stream")
async def ws_translate_stream(ws: WebSocket):
    await ws.accept()
    try:
        data = await _recv_all_binary(ws)
        text, whisper_hint, _ = ASR_MODEL.transcribe(data)
        src = MT.resolve_safe_src(text, whisper_hint)

        fr_text = MT.to_fr(text, src)
        lg_text = MT.to_lg(text, src)

        # Send text first
        await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))
        # Audio header (float32 mono)
        await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))

        def on_chunk(pcm: np.ndarray):
            asyncio.create_task(ws.send_bytes(pcm.tobytes(order="C")))

        try:
            TTS.stream_tts(lg_text, on_chunk=on_chunk, chunk_ms=600)
        except Exception:
            traceback.print_exc()
        finally:
            await ws.send_text("DONE")
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()

# -------------------- CHAT (AUDIO STREAM) --------------------
@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(ws: WebSocket):
    await ws.accept()
    try:
        data = await _recv_all_binary(ws)
        user_text, whisper_hint, _ = ASR_MODEL.transcribe(data)
        src = MT.resolve_safe_src(user_text, whisper_hint)

        # Chat pipeline: to EN → Qwen (FR) → ensure FR → FR→LG
        user_en = MT.to_en(user_text, src)
        qwen_fr = QWEN.chat_fr(user_en, temperature=0.2)
        qwen_fr = _ensure_french(qwen_fr)
        basaa   = MT.to_lg(qwen_fr, "fr")

        # Send reply text first
        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
        # Audio header
        await ws.send_text(json.dumps({"sr": S.tts_sr, "format": "pcm_f32"}))

        def on_chunk(pcm: np.ndarray):
            asyncio.create_task(ws.send_bytes(pcm.tobytes(order="C")))

        try:
            TTS.stream_tts(basaa, on_chunk=on_chunk, chunk_ms=600)
        except Exception:
            traceback.print_exc()
        finally:
            await ws.send_text("DONE")
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()
