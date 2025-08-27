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
        _tts_orpheus = Orpheus(
            os.environ.get("PATH_ORPHEUS", "/data/models/orpheus_bundle"),
            sr_out=24000,
            voice_name=os.environ.get("ORPHEUS_VOICE", "basaa_speaker"),
        )
    return _asr_general, _asr_basaa, _llm_qwen, _mt_m2m, _tts_orpheus

async def send_json(ws: WebSocket, typ: str, **payload):
    data = dict(payload); data["type"] = typ
    await ws.send_text(json.dumps(data, ensure_ascii=False))

@app.get("/")
async def root():
    return {"ok": True}

# --------------------------- helpers ---------------------------
def _norm_lang(s: Optional[str]) -> str:
    s = (s or "fr").lower().strip()
    return "lg" if s in ("lg","bas","basaa") else ("en" if s == "en" else "fr")

def _is_stop(payload: str) -> bool:
    try:
        data = json.loads(payload)
        if isinstance(data, dict) and str(data.get("type","")).lower() == "stop":
            return True
    except Exception:
        pass
    return payload.strip().upper() == "DONE"

def _maybe_lang(payload: str) -> Optional[str]:
    try:
        data = json.loads(payload)
        if isinstance(data, dict) and "lang" in data and isinstance(data["lang"], str):
            return _norm_lang(data["lang"])
    except Exception:
        pass
    return None

# --------------------------- TEXT CHAT (pair) ---------------------------
@app.websocket("/ws/chat_text")
async def ws_chat_text(websocket: WebSocket):
    """
    Client sends:  {"text": "...", "lang": "fr|en|lg"}   // 'lang' optional
    Server replies once with: {"type":"pair","fr":"...","lg":"..."}
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

            user_text = str(data["text"]).strip()
            src = _norm_lang(data.get("lang") or data.get("src"))

            # Prepare a French prompt for Qwen
            if src == "en":
                prompt_fr = llm.en_to_fr(user_text)
            elif src == "lg":
                prompt_fr = m2m.bas_to_fr(user_text)
            else:
                prompt_fr = user_text

            fr = llm.chat_as_fr(prompt_fr)
            lg = m2m.fr_to_bas(fr)

            await send_json(websocket, "pair", fr=fr, lg=lg)
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ------------------------- TEXT TRANSLATE (pair) ------------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(websocket: WebSocket):
    """
    Client sends:  {"text": "...", "src": "fr|en|lg"}    // 'src' or 'lang' optional
    Server replies once with: {"type":"pair","fr":"...","lg":"..."}
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

            src = _norm_lang(data.get("src") or data.get("lang"))
            text = str(data["text"]).strip()

            if src == "en":
                fr = llm.en_to_fr(text)
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

# ----------------------- ONE-SHOT TTS ----------------
@app.websocket("/ws/tts_once")
async def ws_tts_once(websocket: WebSocket):
    """
    Client sends: {"text":"..."} OR raw text; server returns raw WAV bytes once.
    """
    await websocket.accept()
    try:
        *_, tts = services()
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                text = str(data.get("text","")).strip()
            except Exception:
                text = msg.strip()
            if not text:
                await send_json(websocket, "error", message="missing text")
                continue
            wav = tts.tts(text)
            await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ---------------------- AUDIO CHAT (NON-STREAMING) ----------------------
@app.websocket("/ws/audio_chat")
async def ws_audio_chat(websocket: WebSocket):
    """
    Non-streaming audio conversation:
      - client sends PCM16 16k mono frames
      - client sends "DONE" (or {"type":"stop"})
      - server replies:
          * {"fr":"...", "lg":"...", "asr": "..."}   (one JSON block)
          * <binary WAV> (Basaa TTS)
    """
    await websocket.accept()
    buf = bytearray()
    cur_lang = "fr"
    try:
        asr_general, asr_basaa, llm, m2m, tts = services()
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"):
                    buf.extend(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if _is_stop(txt):
                        # process
                        x16 = decode_audio_to_16k_float_mono(bytes(buf))
                        if x16.size == 0:
                            await send_json(websocket, "error", message="empty audio")
                            buf.clear()
                            continue

                        # ASR
                        lang = _norm_lang(cur_lang)
                        if lang in ("fr","en"):
                            asr_text = asr_general.transcribe(x16, lang=lang)
                        else:
                            asr_text = asr_basaa.transcribe(x16)

                        # Build FR
                        if lang == "en":
                            fr_text = llm.en_to_fr(asr_text)
                        elif lang == "lg":
                            # LG input -> converse in FR (Qwen on FR pivot)
                            fr_text = llm.chat_as_fr(m2m.bas_to_fr(asr_text))
                        else:
                            fr_text = llm.chat_as_fr(asr_text)

                        # Basaa text via M2M
                        lg_text = m2m.fr_to_bas(fr_text)

                        # Send pair (client shows Basaa first, then FR after WAV)
                        await websocket.send_text(json.dumps(
                            {"fr": fr_text, "lg": lg_text, "asr": asr_text},
                            ensure_ascii=False
                        ))

                        # TTS -> single WAV
                        wav = tts.tts(lg_text)
                        await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))

                        buf.clear()
                    else:
                        # maybe a language header
                        maybe = _maybe_lang(txt)
                        if maybe: cur_lang = maybe
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ---------------------- TRANSLATE (NON-STREAMING) -----------------------
@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket):
    """
    Non-streaming audio translate:
      - client sends PCM16 16k mono frames
      - client sends "DONE" (or {"type":"stop"})
      - server replies:
          * {"fr":"...", "lg":"...", "asr": "..."}   (one JSON block)
          * <binary WAV> (Basaa TTS of lg)
      Behavior:
        - src=fr: fr=ASR,      lg=fr->lg
        - src=en: fr=en->fr,   lg=fr->lg
        - src=lg: fr=lg->fr,   lg=ASR (echo)
    """
    await websocket.accept()
    buf = bytearray()
    cur_lang = "fr"
    try:
        asr_general, asr_basaa, llm, m2m, tts = services()
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"):
                    buf.extend(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if _is_stop(txt):
                        x16 = decode_audio_to_16k_float_mono(bytes(buf))
                        if x16.size == 0:
                            await send_json(websocket, "error", message="empty audio")
                            buf.clear(); continue

                        lang = _norm_lang(cur_lang)
                        if lang in ("fr","en"):
                            asr_text = asr_general.transcribe(x16, lang=lang)
                        else:
                            asr_text = asr_basaa.transcribe(x16)

                        if lang == "en":
                            fr_text = llm.en_to_fr(asr_text)
                            lg_text = m2m.fr_to_bas(fr_text)
                        elif lang == "fr":
                            fr_text = asr_text
                            lg_text = m2m.fr_to_bas(fr_text)
                        else:  # lg
                            fr_text = m2m.bas_to_fr(asr_text)
                            lg_text = asr_text

                        await websocket.send_text(json.dumps(
                            {"fr": fr_text, "lg": lg_text, "asr": asr_text},
                            ensure_ascii=False
                        ))

                        # TTS
                        wav_src_text = lg_text  # always speak Basaa in translate mode
                        wav = tts.tts(wav_src_text)
                        await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))

                        buf.clear()
                    else:
                        maybe = _maybe_lang(txt)
                        if maybe: cur_lang = maybe
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ---------------------- AUDIO STREAM (OPTIONAL) --------------------
@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(
    websocket: WebSocket,
    mode: str = Query("chat"),     # "chat" | "translate"
    lang: str = Query("fr"),       # "fr" | "en" | "lg"
):
    """
    Streaming contract (kept for later):
      1) {"type":"text_line","lang":"fr","text": FR}
      2) {"type":"pcm_header","sr":24000,"format":"f32le"}
      3) <binary> f32le chunks (80ms)
      4) {"type":"wav"}  +  <binary> full WAV
      5) {"type":"text_line","lang":"lg","text": LG}
      6) {"type":"text_line","lang":"asr","text": ASR}
    """
    await websocket.accept()
    buf = bytearray()
    cur_lang = _norm_lang(lang)
    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"): buf.extend(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if _is_stop(txt):
                        await _handle_stop_stream(websocket, bytes(buf), mode, cur_lang)
                        buf.clear()
                    else:
                        maybe = _maybe_lang(txt)
                        if maybe: cur_lang = maybe
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

    if lang in ("fr","en"): asr_text = asr_general.transcribe(x16, lang=lang)
    else:                   asr_text = asr_basaa.transcribe(x16)

    if (mode or "chat").lower() == "chat":
        fr_text = llm.chat_as_fr(asr_text if lang!="en" else llm.en_to_fr(asr_text))
    else:
        fr_text = llm.en_to_fr(asr_text) if lang=="en" else asr_text

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
