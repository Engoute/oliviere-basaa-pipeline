# FILE: app/main.py
import json, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from .config import S
from .utils_audio import wav_bytes_from_float32
from .asr_whisper import ASR
from .mt_m2m import M2M
from .llm_qwen import QwenAgent
from .tts_orpheus import Orpheus

app = FastAPI(title="Basaa Realtime Pipeline", version="1.0")

# Instantiate models once
ASR_MODEL = ASR(S.path_whisper)      # looks under /data/models/whisper_hf_resolved by default
MT        = M2M(S.path_m2m)          # to_fr(text, src) & to_lg(text, src)
QWEN      = QwenAgent(S.path_qwen)   # FR persona
TTS       = Orpheus(S.path_orpheus, sr_out=S.tts_sr)

def _decide_lang(whisper_text: str, whisper_code: str) -> str:
    # Basaa -> "lg" (no TTS), French/English/other -> Basaa + TTS
    return whisper_code if whisper_code in ("fr", "en") else "lg"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

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
                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception:
        traceback.print_exc()
        await ws.close()
        return

    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    lang = _decide_lang(text, wlang)

    fr_text, lg_text = "", ""
    wav_bytes = None

    if lang == "lg":
        # Input is Basaa -> return FR text only
        fr_text = MT.to_fr(text, "lg")
        lg_text = text
    else:
        # Input is FR/EN/other -> return Basaa + WAV
        fr_text = text if lang == "fr" else MT.to_fr(text, lang)
        lg_text = MT.to_lg(text, lang)
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
                    if msg["text"] == "DONE":
                        break
                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return
    except Exception:
        traceback.print_exc()
        await ws.close()
        return

    # Normalize to French for Qwen
    text, wlang, _ = ASR_MODEL.transcribe(bytes(buf))
    user_fr = MT.to_fr(text, wlang if wlang in ("fr", "en", "lg") else "fr")
    qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)

    # FR -> Basaa (+ WAV)
    basaa = MT.to_lg(qwen_fr, "fr")
    wav = TTS.tts(basaa)
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if wav.size > 0 else None

    await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
    await ws.close()

@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        # Quick Basaa heuristic: special letters likely found in Basaa orthography
        is_basaa = any(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in text.lower())
        out = MT.to_fr(text, "lg") if is_basaa else MT.to_lg(text, "fr")
        await ws.send_text(out)
    except Exception:
        traceback.print_exc()
    finally:
        await ws.close()
