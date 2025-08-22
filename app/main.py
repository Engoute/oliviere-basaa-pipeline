# FILE: app/main.py
import json, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M

app = FastAPI(title="Basaa Whisper+M2M", version="0.2")

ASR_MODEL = ASR(S.path_whisper)
MT        = M2M(S.path_m2m)

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

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

    # Policy:
    # - If input is Basaa (lg) -> output French
    # - If input is FR/EN/other -> output Basaa
    if wlang == "lg":
        fr_text = MT.to_fr(text, "lg")
        lg_text = text
    else:
        fr_text = text if wlang == "fr" else MT.to_fr(text, wlang)
        lg_text = MT.to_lg(text, wlang)

    await ws.send_text(json.dumps({"fr": fr_text, "lg": lg_text}, ensure_ascii=False))
    await ws.close()
