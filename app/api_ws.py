# FastAPI websocket for listening/streaming with f32 PCM and FR-first text event.
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from .utils_audio import decode_audio_to_16k_float_mono, wav_bytes_from_float32
from .asr_whisper import WhisperGeneralASR, WhisperBasaaASR
from .llm_qwen import QwenInstruct
from .mt_m2m import M2MTranslator
from .tts_orpheus import Orpheus

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- lazy singletons (load once) ----------
_asr_general: Optional[WhisperGeneralASR] = None
_asr_basaa: Optional[WhisperBasaaASR] = None
_llm_qwen: Optional[QwenInstruct] = None
_mt_m2m: Optional[M2MTranslator] = None
_tts_orpheus: Optional[Orpheus] = None


def services():
    global _asr_general, _asr_basaa, _llm_qwen, _mt_m2m, _tts_orpheus
    if _asr_general is None:
        _asr_general = WhisperGeneralASR(
            os.environ.get("PATH_WHISPER_GENERAL", "/data/models/whisper_general_resolved")
        )
    if _asr_basaa is None:
        _asr_basaa = WhisperBasaaASR(
            os.environ.get("PATH_WHISPER_BASAA", "/data/models/whisper_hf_resolved")
        )
    if _llm_qwen is None:
        _llm_qwen = QwenInstruct(os.environ.get("PATH_QWEN", "/data/models/qwen2_5_instruct_7b"))
    if _mt_m2m is None:
        _mt_m2m = M2MTranslator(os.environ.get("PATH_M2M", "/data/models/m2m100_1p2B"))
    if _tts_orpheus is None:
        _tts_orpheus = Orpheus(
            os.environ.get("PATH_ORPHEUS", "/data/models/orpheus_bundle"),
            sr_out=24000,
            voice_name=os.environ.get("ORPHEUS_VOICE", "basaa_speaker"),
        )
    return _asr_general, _asr_basaa, _llm_qwen, _mt_m2m, _tts_orpheus


async def send_json(ws: WebSocket, typ: str, **payload):
    data = dict(payload)
    data["type"] = typ
    await ws.send_text(json.dumps(data, ensure_ascii=False))


@app.get("/")
async def root():
    return {"ok": True, "ws": "/ws/audio_chat_stream"}


# --------------------------- MAIN WS (matches client) ---------------------------
@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(
    websocket: WebSocket,
    mode: str = Query("chat"),     # "chat" or "translate"
    lang: str = Query("fr"),       # "fr", "en", "lg"
):
    """
    Client protocol (aligned with your RealtimeTranslator):
      - client sends raw PCM16 chunks (binary) @16k mono
      - when done, client sends {"type":"stop"} as text
      - server replies, in this order:
          1) {"type":"text_line","lang":"fr","text": "..."}        # FR first
          2) {"type":"pcm_header","sr":24000,"format":"f32le"}     # streaming format
          3) <binary> f32le PCM chunks (multiple)                   # onPcmChunk
          4) {"type":"wav"}  +  <binary> full WAV (one frame)       # onWav
          5) {"type":"text_line","lang":"lg","text":"..."}          # Basaa line
          6) {"type":"text_line","lang":"asr","text":"..."}         # user transcript
    """
    await websocket.accept()
    buf = bytearray()

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"):
                    buf.extend(msg["bytes"])
                elif msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        data = {"type": msg["text"]}
                    if data.get("type") == "stop":
                        await _handle_stop(websocket, bytes(buf), mode, lang)
                        buf.clear()
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))


# ------------------------------- stop handler ----------------------------------
async def _handle_stop(ws: WebSocket, payload: bytes, mode: str, lang: str):
    asr_general, asr_basaa, llm, m2m, tts = services()

    # 1) decode to 16k float mono
    x16 = decode_audio_to_16k_float_mono(payload)  # float32 [-1,1], sr=16k
    if x16.size == 0:
        await send_json(ws, "error", message="empty audio")
        return

    # 2) ASR
    lang_l = (lang or "fr").lower()
    if lang_l in ("fr", "en"):
        asr_text = asr_general.transcribe(x16, lang=lang_l)
    else:
        asr_text = asr_basaa.transcribe(x16)  # basaa

    # 3) Build FR text based on mode
    mode_l = (mode or "chat").lower()
    if mode_l == "chat":
        # Conversational: Qwen produces FR reply directly
        fr_text = llm.chat_as_fr(asr_text)
    else:
        # Translation: show FR text first
        if lang_l == "en":
            # Prefer a robust EN->FR bridge (use your helper or swap to m2m.en_to_fr)
            fr_text = llm.en_to_fr(asr_text)
        else:
            fr_text = asr_text  # already FR

    # 3b) Emit FR line FIRST
    await send_json(ws, "text_line", lang="fr", text=fr_text)

    # 4) FR -> Basaa
    lg_text = m2m.fr_to_bas(fr_text)

    # 5) High-quality “streaming”: synth once, slice to f32le chunks
    sr_out = tts.sr_out  # typically 24000
    await send_json(ws, "pcm_header", sr=sr_out, format="f32le")

    # Generate full waveform, then slice (Orpheus quality matches one-shot)
    wav_f32 = tts.tts(lg_text)  # float32 mono [-1..1] @ sr_out
    if wav_f32.size == 0:
        await send_json(ws, "error", message="tts returned empty audio")
        # still send remaining text so UI completes
        await send_json(ws, "text_line", lang="lg", text=lg_text)
        await send_json(ws, "text_line", lang="asr", text=asr_text)
        return

    # Stream ~80ms chunks as f32le
    step = max(1, int(sr_out * 0.080))
    f32 = wav_f32.astype("<f4", copy=False)  # little-endian
    for i in range(0, len(f32), step):
        chunk = f32[i : i + step]
        if chunk.size:
            # websockets require bytes; use the raw view
            await ws.send_bytes(chunk.tobytes())

    # Send full WAV for replay (announce with a marker first)
    wav_bytes = wav_bytes_from_float32(wav_f32, sr_out)
    await send_json(ws, "wav")
    await ws.send_bytes(wav_bytes)

    # 6) Finish with Basaa line then ASR transcript
    await send_json(ws, "text_line", lang="lg", text=lg_text)
    await send_json(ws, "text_line", lang="asr", text=asr_text)
