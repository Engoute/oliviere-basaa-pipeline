# app/api_ws.py
from __future__ import annotations
import asyncio, json, os
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
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- lazy singletons ----------
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
        _tts_orpheus = Orpheus(os.environ.get("PATH_ORPHEUS", "/data/models/orpheus_bundle"),
                               sr_out=24000,
                               voice_name=os.environ.get("ORPHEUS_VOICE", "basaa_speaker"))
    return _asr_general, _asr_basaa, _llm_qwen, _mt_m2m, _tts_orpheus

async def send_json(ws: WebSocket, typ: str, **payload):
    data = dict(payload); data["type"] = typ
    await ws.send_text(json.dumps(data, ensure_ascii=False))

@app.get("/")
async def root():
    return {"ok": True}

# --------------------------- TEXT: chat (pair) ---------------------------
@app.websocket("/ws/chat_text")
async def ws_chat_text(websocket: WebSocket,
                       lang: str = Query("fr")):
    """
    Client sends:  {"text": "..."}
    Server replies once with: {"type":"pair","fr":"...","lg":"..."}
    (FR is Qwen's output; LG is M2M(fr->lg))
    """
    await websocket.accept()
    try:
        asr_g, asr_lg, llm, m2m, _ = services()
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if not isinstance(data, dict) or "text" not in data:
                await send_json(websocket, "error", message="bad payload")
                continue

            user_text = str(data["text"]).strip()
            lang_l = (lang or "fr").lower()

            # Prepare a French prompt for Qwen
            if lang_l == "en":
                prompt_fr = m2m.en_to_fr(user_text)
            elif lang_l == "lg":
                prompt_fr = m2m.bas_to_fr(user_text)
            else:
                prompt_fr = user_text

            fr = llm.chat_as_fr(prompt_fr)          # <- THIS is what you want to show on toggle
            lg = m2m.fr_to_bas(fr)

            await send_json(websocket, "pair", fr=fr, lg=lg)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ----------------------- TEXT: translate (pair) -------------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(websocket: WebSocket,
                            lang: str = Query("fr")):
    """
    Client sends:  {"text": "..."}
    Server replies once with: {"type":"pair","fr":"...","lg":"..."}
      - If lang in {fr,en}: fr = pivot(FR or EN->FR), lg = M2M(fr->lg)
      - If lang == lg:      fr = M2M(lg->fr),       lg = original (echo)
    """
    await websocket.accept()
    try:
        _, _, llm, m2m, _ = services()
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if not isinstance(data, dict) or "text" not in data:
                await send_json(websocket, "error", message="bad payload")
                continue

            src = (lang or "fr").lower()
            text = str(data["text"]).strip()

            if src == "en":
                fr = llm.en_to_fr(text)      # short, robust EN->FR bridge
                lg = m2m.fr_to_bas(fr)
            elif src == "fr":
                fr = text
                lg = m2m.fr_to_bas(fr)
            else:  # src == "lg"
                fr = m2m.bas_to_fr(text)
                lg = text

            await send_json(websocket, "pair", fr=fr, lg=lg)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ----------------------- ONE-SHOT TTS (Lire à voix haute) ----------------
@app.websocket("/ws/tts_once")
async def ws_tts_once(websocket: WebSocket):
    """
    Client sends: {"text":"..."}; server returns raw WAV bytes once.
    """
    await websocket.accept()
    try:
        *_, tts = services()
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if "text" not in data:
                await send_json(websocket, "error", message="missing text")
                continue
            text = str(data["text"]).strip()
            wav = tts.tts(text)
            await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ---------------------- AUDIO STREAM (listening page) --------------------
@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(
    websocket: WebSocket,
    mode: str = Query("chat"),     # "chat" | "translate"
    lang: str = Query("fr"),       # "fr" | "en" | "lg"
):
    """
    Same streaming contract we discussed: FR line first, then f32 PCM stream, then full WAV,
    then Basaa line and ASR last.
    """
    await websocket.accept()
    buf = bytearray()
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"): buf.extend(msg["bytes"])
                elif msg.get("text"):
                    try: data = json.loads(msg["text"])
                    except Exception: data = {"type": msg["text"]}
                    if data.get("type") == "stop":
                        await _handle_stop_stream(websocket, bytes(buf), mode, lang)
                        buf.clear()
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

async def _handle_stop_stream(ws: WebSocket, payload: bytes, mode: str, lang: str):
    asr_general, asr_basaa, llm, m2m, tts = services()

    x16 = decode_audio_to_16k_float_mono(payload)
    if x16.size == 0:
        await send_json(ws, "error", message="empty audio")
        return

    lang_l = (lang or "fr").lower()
    if lang_l in ("fr","en"): asr_text = asr_general.transcribe(x16, lang=lang_l)
    else:                      asr_text = asr_basaa.transcribe(x16)

    mode_l = (mode or "chat").lower()
    if mode_l == "chat":
        fr_text = llm.chat_as_fr(asr_text if lang_l!="en" else llm.en_to_fr(asr_text))
    else:
        fr_text = llm.en_to_fr(asr_text) if lang_l=="en" else asr_text

    await send_json(ws, "text_line", lang="fr", text=fr_text)

    lg_text = m2m.fr_to_bas(fr_text)

    await send_json(ws, "pcm_header", sr=tts.sr_out, format="f32le")
    wav_f32 = tts.tts(lg_text)
    step = max(1, int(tts.sr_out * 0.080))
    f32 = wav_f32.astype("<f4", copy=False)
    for i in range(0, len(f32), step):
        await ws.send_bytes(f32[i:i+step].tobytes())

    wav_bytes = wav_bytes_from_float32(wav_f32, tts.sr_out)
    await send_json(ws, "wav")
    await ws.send_bytes(wav_bytes)

    await send_json(ws, "text_line", lang="lg", text=lg_text)
    await send_json(ws, "text_line", lang="asr", text=asr_text)
