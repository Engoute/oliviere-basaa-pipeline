# FILE: app/api_ws.py
import asyncio
import base64
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .pipeline_core import load_models, run_pipeline, labeled_block

app = FastAPI()
MODELS = load_models()  # warm-load at startup


@app.get("/healthz")
async def healthz():
    return JSONResponse({"ok": True})


@app.websocket("/ws/listen")
async def ws_listen(ws: WebSocket):
    """
    Protocol (text JSON vs binary):
      -> client sends JSON {"type":"init","mode":"chat"|"translate","lang":"fr"|"en"|"lg"}
      -> client streams raw PCM16LE @16k as BINARY frames
      -> client sends JSON {"type":"end"}

      <- server sends JSON {"type":"pcm_header","sr": <int>, "format":"pcm16le"}
      <- server streams BINARY PCM16LE chunks at sr
      <- server sends JSON {"type":"text","payload": "<Vous:/FR:/Basaa: block>"}
      <- server sends JSON {"type":"wav","bytes_b64": "<base64 PCM16 wav>"}
      <- server sends JSON {"type":"done"}
    """
    await ws.accept()
    mode: Optional[str] = None
    lang: Optional[str] = None
    buf = bytearray()

    try:
        # Expect init first (text frame)
        msg = await ws.receive_text()
        import json
        js = json.loads(msg)
        if not (isinstance(js, dict) and js.get("type") == "init"):
            await ws.close(code=1002)
            return
        mode = js.get("mode", "chat")
        lang = js.get("lang", "fr")
    except Exception:
        await ws.close(code=1002)
        return

    # Receive frames
    try:
        while True:
            data = await ws.receive()
            if "text" in data:
                import json
                js = json.loads(data["text"])
                if js.get("type") == "end":
                    break
                # ignore other text frames
            elif "bytes" in data:
                # raw PCM16LE bytes from device (16k)
                buf.extend(data["bytes"])
    except WebSocketDisconnect:
        return

    # Run the pipeline
    try:
        asr, fr, lg, wav_f32, sr_out = run_pipeline(MODELS, mode, lang, bytes(buf))
    except Exception as e:
        # best-effort error surface
        await ws.send_json({"type": "error", "message": f"{e}"})
        await ws.close()
        return

    # Stream TTS as PCM16LE (header + chunks)
    await ws.send_json({"type": "pcm_header", "sr": int(sr_out), "format": "pcm16le", "channels": 1})

    # Slice ~120 ms chunks
    step = max(1, int(sr_out * 0.120))
    import numpy as np
    pcm16 = (np.clip(wav_f32, -1.0, 1.0) * 32767.0).astype("<i2")
    raw = pcm16.tobytes(order="C")
    for i in range(0, len(raw), step * 2):
        await ws.send_bytes(raw[i:i + step * 2])

    # Labeled text block in the order your UI expects to fill lanes
    await ws.send_json({"type": "text", "payload": labeled_block(asr, fr, lg)})

    # Also ship a full WAV for replay
    from .utils_audio import wav_bytes_from_float32
    wav_bytes = wav_bytes_from_float32(wav_f32, sr=int(sr_out))
    await ws.send_json({"type": "wav", "bytes_b64": base64.b64encode(wav_bytes).decode("ascii")})

    # done
    await ws.send_json({"type": "done"})
    await ws.close()
