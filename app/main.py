# FILE: app/main.py
from .fast import speed_tweaks
speed_tweaks()

import json, traceback
from os import environ
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M
from .tts_orpheus import Orpheus
from .llm_qwen import QwenAgent
from .utils_audio import wav_bytes_from_float32

app = FastAPI(title="Basaa Realtime Pipeline", version="0.6")

# ---------- Resolve Orpheus acoustic root exactly like in Colab ----------
def _pick_orpheus_acoustic() -> Path:
    """
    Prefer symlinks created by bootstrap.py, then fall back to S.path_orpheus.
    If a bundle root is passed, auto-select its 'acoustic_model' child.
    """
    candidates = [
        Path("/data/models/orpheus_resolved"),     # generic symlink
        Path("/data/models/orpheus_3b_resolved"),  # your original symlink
        Path(S.path_orpheus),                      # env/Config fallback
    ]
    for p in candidates:
        if p.exists():
            root = p
            break
    else:
        root = Path(S.path_orpheus)

    # If user gives the bundle root, use its acoustic folder
    if (root / "acoustic_model").exists():
        return root / "acoustic_model"
    return root

# ---------- Model init ----------
ASR_MODEL = ASR(S.path_whisper)
MT        = M2M(S.path_m2m)

_acoustic_dir = _pick_orpheus_acoustic()
TTS       = Orpheus(str(_acoustic_dir), sr_out=S.tts_sr)

QWEN      = QwenAgent(S.path_qwen)

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# ---------- Endpoints ----------
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
        if getattr(wav, "size", 0) > 0:
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
    wav_bytes = wav_bytes_from_float32(wav, S.tts_sr) if getattr(wav, "size", 0) > 0 else None

    # 5) Return JSON + single WAV frame
    await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    if wav_bytes:
        await ws.send_bytes(wav_bytes)
    await ws.close()
