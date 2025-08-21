# /workspace/app/main.py
import asyncio, json, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from .config import S
from .utils_audio import pcm16le_bytes_to_float32_mono, resample, wav_bytes_from_float32
from .asr_whisper import ASR
from .mt_m2m import M2M
from .llm_qwen import QwenAgent
from .tts_orpheus import Orpheus

app = FastAPI(title="Basaa Realtime Pipeline", version="1.0")

# ---- init models (GPU) ----
# NOTE: ASR prefers /data/models/whisper_hf_resolved automatically (from bootstrap).
ASR_MODEL = ASR(S.path_whisper)
MT = M2M(S.path_m2m)
QWEN = QwenAgent(S.path_qwen)
TTS = Orpheus(S.path_orpheus, sr_out=S.tts_sr)

def decide_lang(text: str, whisper_code: str) -> str:
    """
    Very simple rule:
      - if Whisper says 'fr' or 'en' -> respect it;
      - otherwise treat as Basaa ('lg').
    """
    if whisper_code in ("fr", "en"):
        return whisper_code
    return "lg"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# 1) DIRECT TRANSLATION (audio → text + optional WAV)
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
                else:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception:
        traceback.print_exc()
        await ws.close()
        return

    # ASR
    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    lang = decide_lang(text, wlang)

    # Decide outputs
    fr_text, lg_text = "", ""
    wav_bytes = None

    if lang == "lg":
        # Basaa input → French output (no TTS)
        fr_text = MT.to_fr(text, "lg")
        lg_text = text  # echo original
    else:
        # FR/EN/other → Basaa (with TTS)
        fr_text = text if lang == "fr" else MT.to_fr(text, lang)
        lg_text = MT.to_lg(text, lang)
        wav = TTS.tts(lg_text)
        if wav.size > 0:
            wav_bytes = wav_bytes_from_float32(wav, S.tts_sr)

    # Send JSON
    await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))

    # Send WAV if we have it (FR/EN → Basaa case)
    if wav_bytes:
        await ws.send_bytes(wav_bytes)

    await ws.close()

# 2) AUDIO CHAT WITH QWEN (audio → Qwen EN/FR → Basaa + WAV)
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
                else:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception:
        traceback.print_exc()
        await ws.close()
        return

    # ASR
    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    lang = decide_lang(text, wlang)

    # Normalize user to FR for Qwen (your policy)
    user_fr = MT.to_fr(text, lang)

    # Qwen answers in FR (persona hides pipeline)
    qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)

    # Translate FR → Basaa for the user
    basaa = MT.to_lg(qwen_fr, "fr")

    # TTS (Basaa)
    wav = TTS.tts(basaa)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    # Send JSON + audio
    await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
    await ws.close()

# 3) TEXT-ONLY translation quick WS (client sends text; returns Basaa or French)
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        recv = await ws.receive_text()
        text = recv.strip()

        # Heuristic language check (diacritics/letters common in Basaa)
        is_basaa = sum(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in text.lower()) >= 1

        if is_basaa:
            # Basaa → French
            out = MT.to_fr(text, "lg")
        else:
            # Assume FR (or EN → FR first inside MT.to_lg if you implemented it that way)
            out = MT.to_lg(text, "fr")

        await ws.send_text(out)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()
