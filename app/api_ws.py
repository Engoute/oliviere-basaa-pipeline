# FastAPI websocket for listening/streaming with f32 PCM and FR-first text event.
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from .utils_audio import decode_audio_to_16k_float_mono, wav_bytes_from_float32
from .asr_whisper import WhisperGeneralASR, WhisperBasaaASR
from .llm_qwen import QwenInstruct
from .mt_m2m import M2MTranslator
from .tts_orpheus import Orpheus

import numpy as np
import os

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
        _asr_general = WhisperGeneralASR(os.environ.get("PATH_WHISPER_GENERAL", "/data/models/whisper_general_resolved"))
    if _asr_basaa is None:
        _asr_basaa = WhisperBasaaASR(os.environ.get("PATH_WHISPER_BASAA", "/data/models/whisper_hf_resolved"))
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

# Small helper for consistent JSON events
async def send_json(ws: WebSocket, typ: str, **payload):
    msg = payload.copy()
    msg["type"] = typ
    await ws.send_text(json.dumps(msg, ensure_ascii=False))

# ---------- ROUTES ----------
@app.get("/")
async def root():
    return {"ok": True, "ws": "/ws/listen"}

@app.websocket("/ws/listen")
async def ws_listen(
    websocket: WebSocket,
    mode: str = Query("chat"),     # "chat" or "translate"
    lang: str = Query("fr"),       # "fr", "en", "lg"
):
    """
    Protocol (compatible with your client):
      - client sends raw PCM16 chunks (binary) captured @16k mono
      - when done, client sends text message: {"type":"stop"}
      - server replies:
          * {"type":"text_line","lang":"fr","text": "..."}  (Qwen output FIRST)
          * {"type":"pcm_header","sr":24000,"format":"f32le"}
          * binary frames of raw float32 little-endian mono @ sr (pcm_chunk)
          * {"type":"wav","bytes_b64": "..."}  (final full wav for replay)
          * {"type":"text_line","lang":"lg","text":"..."}    (Basaa line)
          * {"type":"text_line","lang":"asr","text":"..."}   (user transcript)
    """
    await websocket.accept()
    buf = bytearray()
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if "bytes" in msg and msg["bytes"] is not None:
                    # accumulate raw PCM16 or WAV etc (client sends PCM16 16k)
                    buf.extend(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        data = {"type": msg["text"]}
                    if data.get("type") == "stop":
                        await handle_stop(websocket, bytes(buf), mode, lang)
                        buf.clear()
                    # ignore other client control messages
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))


# ---------- main stop handler ----------
async def handle_stop(ws: WebSocket, payload: bytes, mode: str, lang: str):
    asr_general, asr_basaa, llm, m2m, tts = services()

    # 1) decode/normalize to 16k float mono
    x16 = decode_audio_to_16k_float_mono(payload)  # float32 [-1,1], sr=16k
    if x16.size == 0:
        await send_json(ws, "error", message="empty audio")
        return

    # 2) ASR (FR/EN use general; LG uses finetuned basaa)
    if lang.lower() in ("fr", "en"):
        asr_text = asr_general.transcribe(x16, lang=lang.lower())
    else:
        asr_text = asr_basaa.transcribe(x16)  # language is basaa

    # 3) Build FR text depending on mode
    #    - chat: Qwen generates FR response from ASR text
    #    - translate: FR (or EN) -> we forward to M2M, but we also want to DISPLAY the FR text FIRST
    if mode.lower() == "chat":
        fr_text = llm.chat_as_fr(asr_text)  # your helper that returns a single FR sentence/paragraph
    else:
        # "translate" mode: if EN, optionally paraphrase to FR once; if FR already, pass-through.
        if lang.lower() == "en":
            fr_text = llm.en_to_fr(asr_text)  # short robust bridge (keeps style)
        else:
            fr_text = asr_text  # already FR

    # >>> NEW: emit FR line BEFORE any translation / audio <<<
    await send_json(ws, "text_line", lang="fr", text=fr_text)

    # 4) FR -> Basaa
    lg_text = m2m.fr_to_bas(fr_text)

    # 5) TTS: generate once (full wav), then stream f32le chunks (high quality)
    sr_out = tts.sr_out  # usually 24000 from Orpheus
    chunks = []          # keep slices so we can also return the full WAV
    await send_json(ws, "pcm_header", sr=sr_out, format="f32le")

    def on_chunk(f32: np.ndarray):
        # Send raw float32 little-endian bytes (mono)
        chunks.append(f32.copy())
        try:
            ws.loop.create_task(ws.send_bytes(f32.astype("<f4", copy=False).tobytes()))
        except RuntimeError:
            # fallback when loop attribute not available (uvicorn recent)
            asyncio.create_task(ws.send_bytes(f32.astype("<f4", copy=False).tobytes()))

    # Generate once, then slice and stream
    tts.stream_tts(lg_text, on_chunk=on_chunk, chunk_ms=80)

    # Assemble WAV for replay on the client
    if chunks:
        wav = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        wav_bytes = wav_bytes_from_float32(wav, sr_out)
        # We keep raw bytes (no base64) because your client already handles that
        await ws.send_bytes(wav_bytes)            # optional: if your client expects a "wav" binary frame
        await send_json(ws, "wav_ready", length=len(wav_bytes))

    # 6) Emit Basaa line and the user transcript last (your UI displays in that order)
    await send_json(ws, "text_line", lang="lg", text=lg_text)
    await send_json(ws, "text_line", lang="asr", text=asr_text)
