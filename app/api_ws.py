from __future__ import annotations
import asyncio, json, os, io
from typing import Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .utils_audio import decode_audio_to_16k_float_mono, wav_bytes_from_float32
# restore legacy wrappers + keep unified ASR available
from .asr_whisper import WhisperGeneralASR, WhisperBasaaASR
from .llm_qwen import QwenInstruct
from .mt_m2m import M2MTranslator
from .tts_orpheus import Orpheus
from .vision_llava import LLaVAVideo

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
_llava: Optional[LLaVAVideo] = None

def services():
    global _asr_general, _asr_basaa, _llm_qwen, _mt_m2m, _tts_orpheus, _llava
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
            sr_out=24000,                                      # ← lock 24 kHz
            voice_name=os.environ.get("ORPHEUS_VOICE", "basaa_speaker"),
        )
    if _llava is None:
        _llava = LLaVAVideo(os.environ.get("PATH_LLAVA_VIDEO", "/data/models/llava_next_video_resolved"))
    return _asr_general, _asr_basaa, _llm_qwen, _mt_m2m, _tts_orpheus, _llava

async def send_json(ws: WebSocket, typ: str, **payload):
    data = dict(payload); data["type"] = typ
    await ws.send_text(json.dumps(data, ensure_ascii=False))

@app.get("/")
async def root():
    return {"ok": True}

# --------------------------- helpers ---------------------------
def _norm_lang(s: Optional[str]) -> str:
    s = (s or "fr").lower().strip()
    if s in ("lg","bas","basaa"): return "lg"
    if s == "en": return "en"
    return "fr"

def _extract_lang_any(obj: dict) -> Optional[str]:
    # honor nested asr.language / asr.lang / asr.src and top-level lang/src
    if not isinstance(obj, dict): return None
    for k in ("language","lang","src"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return _norm_lang(v)
    asr = obj.get("asr")
    if isinstance(asr, dict):
        for k in ("language","lang","src"):
            v = asr.get(k)
            if isinstance(v, str) and v.strip():
                return _norm_lang(v)
    return None

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
        if isinstance(data, dict):
            got = _extract_lang_any(data)
            if got: return got
    except Exception:
        pass
    return None

def _maybe_text(payload: str) -> Optional[str]:
    try:
        data = json.loads(payload)
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            t = data["text"].strip()
            if t:
                return t
    except Exception:
        pass
    return None

# --------------------------- TEXT CHAT (pair) ---------------------------
@app.websocket("/ws/chat_text")
async def ws_chat_text(websocket: WebSocket):
    await websocket.accept()
    try:
        _, _, llm, m2m, _, _ = services()
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if not isinstance(data, dict) or "text" not in data:
                await send_json(websocket, "error", message="bad payload")
                continue

            user_text = str(data["text"]).strip()
            src = _extract_lang_any(data) or _norm_lang(data.get("lang") or data.get("src"))

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
    await websocket.accept()
    try:
        _, _, llm, m2m, _, _ = services()
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if not isinstance(data, dict) or "text" not in data:
                await send_json(websocket, "error", message="bad payload")
                continue

            src = _extract_lang_any(data) or _norm_lang(data.get("src") or data.get("lang"))
            text = str(data["text"]).strip()

            if src == "en":
                fr = llm.en_to_fr(text)
                lg = m2m.fr_to_bas(fr)
            elif src == "fr":
                fr = text
                lg = m2m.fr_to_bas(fr)
            else:
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
    await websocket.accept()
    try:
        *_, tts, _ = services()
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
            wav = tts.tts(text)  # already adds safety pad
            await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ---------------------- AUDIO CHAT (NON-STREAMING) ----------------------
@app.websocket("/ws/audio_chat")
async def ws_audio_chat(websocket: WebSocket):
    """
    Accepts optional {"text":"..."} override and nested {"asr":{"language":".."}} pins.
    """
    await websocket.accept()
    buf = bytearray()
    cur_lang = "fr"
    typed_text: Optional[str] = None

    try:
        asr_general, asr_basaa, llm, m2m, tts, _ = services()
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"):
                    buf.extend(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if _is_stop(txt):
                        # decide input: typed override OR ASR
                        if typed_text and typed_text.strip():
                            user_text = typed_text.strip()
                            lang = _norm_lang(cur_lang)
                            if lang == "en":
                                prompt_fr = llm.en_to_fr(user_text)
                            elif lang == "lg":
                                prompt_fr = m2m.bas_to_fr(user_text)
                            else:
                                prompt_fr = user_text
                        else:
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
                                prompt_fr = llm.en_to_fr(asr_text)
                            elif lang == "lg":
                                prompt_fr = m2m.bas_to_fr(asr_text)
                            else:
                                prompt_fr = asr_text

                        fr_text = llm.chat_as_fr(prompt_fr)
                        lg_text = m2m.fr_to_bas(fr_text)

                        await websocket.send_text(json.dumps(
                            {"fr": fr_text, "lg": lg_text},
                            ensure_ascii=False
                        ))

                        wav = tts.tts(lg_text)  # full length + tail pad
                        await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))

                        buf.clear(); typed_text = None
                    else:
                        # allow nested asr.language pins and text overrides
                        try:
                            obj = json.loads(txt)
                            maybe = _extract_lang_any(obj)
                            if maybe:
                                cur_lang = maybe
                            t = obj.get("text")
                            if isinstance(t, str) and t.strip():
                                typed_text = t.strip()
                        except Exception:
                            maybe = _maybe_lang(txt)
                            if maybe:
                                cur_lang = maybe
                            else:
                                t = _maybe_text(txt)
                                if t: typed_text = t
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
    Also accepts a TEXT override and nested asr.language pins.
    """
    await websocket.accept()
    buf = bytearray()
    cur_lang = "fr"
    typed_text: Optional[str] = None

    try:
        asr_general, asr_basaa, llm, m2m, tts, _ = services()
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("bytes"):
                    buf.extend(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if _is_stop(txt):
                        lang = _norm_lang(cur_lang)

                        if typed_text and typed_text.strip():
                            asr_text = typed_text.strip()
                        else:
                            x16 = decode_audio_to_16k_float_mono(bytes(buf))
                            if x16.size == 0:
                                await send_json(websocket, "error", message="empty audio")
                                buf.clear(); typed_text = None; continue
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
                        else:
                            fr_text = m2m.bas_to_fr(asr_text)
                            lg_text = asr_text

                        await websocket.send_text(json.dumps(
                            {"fr": fr_text, "lg": lg_text, "asr": asr_text},
                            ensure_ascii=False
                        ))

                        wav = tts.tts(lg_text)  # full length + tail pad
                        await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))

                        buf.clear(); typed_text = None
                    else:
                        try:
                            obj = json.loads(txt)
                            maybe = _extract_lang_any(obj)
                            if maybe: cur_lang = maybe
                            t = obj.get("text")
                            if isinstance(t, str) and t.strip():
                                typed_text = t.strip()
                        except Exception:
                            maybe = _maybe_lang(txt)
                            if maybe: cur_lang = maybe
                            else:
                                t = _maybe_text(txt)
                                if t: typed_text = t
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))

# ---------------------- VISION (ONE-SHOT) -------------------------------
@app.websocket("/ws/vision_once")
async def ws_vision_once(websocket: WebSocket):
    """
    Contract:
      - client sends: {"lang":"fr|en|lg"}  (optional, default fr)
      - client sends PCM16 16k chunks (speech)
      - client sends {"vision":true,"task":"describe"}   (hint)
      - client sends a single binary JPEG snapshot (no UI freeze on client)
      - client sends "DONE"
      - server replies:
           * {"fr":"...", "lg":"...", "asr":"..."}   (French is LLaVA output)
           * <binary WAV> (Basaa TTS of lg)
    """
    await websocket.accept()
    audio_buf = bytearray()
    image_bytes: Optional[bytes] = None
    want_image_next = False
    cur_lang = "fr"

    try:
        asr_general, asr_basaa, llm, m2m, tts, llava = services()

        while True:
            msg = await websocket.receive()
            typ = msg["type"]

            if typ == "websocket.receive":
                if msg.get("bytes"):
                    if want_image_next and image_bytes is None:
                        image_bytes = msg["bytes"]
                        want_image_next = False
                    else:
                        audio_buf.extend(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if _is_stop(txt):
                        # PROCESS
                        x16 = decode_audio_to_16k_float_mono(bytes(audio_buf))
                        if x16.size == 0:
                            await send_json(websocket, "error", message="empty audio")
                            audio_buf.clear()
                            continue

                        lang = _norm_lang(cur_lang)
                        if lang in ("fr","en"):
                            asr_text = asr_general.transcribe(x16, lang=lang)
                            question_fr = asr_text
                        else:
                            asr_text = asr_basaa.transcribe(x16)
                            question_fr = m2m.bas_to_fr(asr_text)

                        if not image_bytes:
                            await send_json(websocket, "error", message="missing image")
                            audio_buf.clear()
                            continue
                        try:
                            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        except Exception:
                            await send_json(websocket, "error", message="bad image")
                            audio_buf.clear()
                            continue

                        try:
                            fr_text = llava.describe_image(img, question_fr)
                        except Exception as e:
                            await send_json(websocket, "error", message=f"vision_failed: {type(e).__name__}: {e}")
                            return

                        lg_text = m2m.fr_to_bas(fr_text)

                        await websocket.send_text(json.dumps(
                            {"fr": fr_text, "lg": lg_text, "asr": asr_text},
                            ensure_ascii=False
                        ))

                        wav = tts.tts(lg_text)
                        await websocket.send_bytes(wav_bytes_from_float32(wav, tts.sr_out))

                        audio_buf.clear()
                        image_bytes = None
                    else:
                        try:
                            data = json.loads(txt)
                            maybe = _extract_lang_any(data)
                            if maybe:
                                cur_lang = maybe
                            elif isinstance(data, dict) and data.get("vision") is True:
                                want_image_next = True
                        except Exception:
                            maybe = _maybe_lang(txt)
                            if maybe:
                                cur_lang = maybe

            elif typ == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        return
    except Exception as e:
        await send_json(websocket, "error", message=str(e))
