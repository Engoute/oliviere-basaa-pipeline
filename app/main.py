# FILE: app/main.py
import json, traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from .config import S
from .asr_whisper import ASR

app = FastAPI(title="Basaa Whisper-Only", version="0.1")

ASR_MODEL = ASR(S.path_whisper)

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

    text, lang, conf = ASR_MODEL.transcribe(bytes(buf))
    await ws.send_text(json.dumps({"text": text, "lang": lang, "lang_conf": conf}, ensure_ascii=False))
    await ws.close()
