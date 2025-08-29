# FILE: app/main.py
from __future__ import annotations

# --- IMPORTANT: block torchvision before transformers loads anywhere ---
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from .fast import speed_tweaks
speed_tweaks()

import asyncio
import json
import re
import unicodedata
import traceback
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState

from .config import S
from .asr_whisper import ASR
from .mt_m2m import M2M
from .tts_orpheus import Orpheus
from .llm_qwen import QwenAgent
from .utils_audio import wav_bytes_from_float32

# --- extra imports for vision ---
import io
from typing import List, Optional
from PIL import Image
import av  # PyAV for video parsing
import torch

app = FastAPI(title="Basaa Realtime Pipeline", version="0.95")

# ---- model singletons --------------------------------------------------------
print(f"[main] Initializing models…")

# ASR dual: basaa + general
print(f"[main]  PATH_WHISPER_BASAA  = {S.path_whisper_basaa}")
print(f"[main]  PATH_WHISPER_GENERAL= {S.path_whisper_general or '(none)'}")
ASR_MODEL = ASR(S.path_whisper_basaa or S.path_whisper, S.path_whisper_general or None)

print(f"[main]  PATH_M2M     = {S.path_m2m}")
MT        = M2M(S.path_m2m)

print(f"[main]  PATH_ORPHEUS  = {S.path_orpheus}")
TTS       = Orpheus(S.path_orpheus, sr_out=S.tts_sr)

print(f"[main]  PATH_QWEN     = {S.path_qwen}")
QWEN      = QwenAgent(S.path_qwen)

# -------- LLaVA-NeXT-Video (local-only) ----------
class _LLaVAVideo:
    STRICT_FRENCH_INSTRUCTION = (
        "Tu es un assistant de vision. Réponds UNIQUEMENT en français, "
        "en 1 à 2 phrases courtes, sans métadonnées ni balises. "
        "Décris uniquement ce qui est VISIBLE. "
        "N'invente pas de détails. Si tu n'es pas sûr, réponds: «Je ne suis pas sûr.»"
    )

    def __init__(self, local_dir: str, device: Optional[str] = None):
        from transformers import AutoProcessor, AutoTokenizer, LlavaNextVideoForConditionalGeneration

        self.local_dir = local_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True

        self.AutoProcessor = AutoProcessor
        self.AutoTokenizer = AutoTokenizer
        self.ModelClass = LlavaNextVideoForConditionalGeneration

        self.processor = self.AutoProcessor.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or self.AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        self._maybe_load_chat_template(self.tokenizer, local_dir)

        self.model = self.ModelClass.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True,
            torch_dtype=self.dtype, device_map="auto"
        ).eval()

    @staticmethod
    def _maybe_load_chat_template(tok, root: str):
        if getattr(tok, "chat_template", None):
            return
        try:
            p = os.path.join(root, "chat_template.json")
            if os.path.isfile(p):
                data = json.load(open(p, "r", encoding="utf-8"))
                tmpl = data.get("chat_template") or data.get("template") or (data if isinstance(data, str) else None)
                if isinstance(tmpl, dict):
                    tmpl = tmpl.get("template")
                if isinstance(tmpl, str) and tmpl.strip():
                    tok.chat_template = tmpl
        except Exception:
            pass

    def _build_prompt(self, question_fr: str) -> str:
        conv = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": f"{self.STRICT_FRENCH_INSTRUCTION}\n\nQuestion: {question_fr}"},
            ],
        }]]
        apply = getattr(self.tokenizer, "apply_chat_template", None) or getattr(self.processor, "apply_chat_template", None)
        if apply:
            try:
                return apply(conv, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return apply(conv, add_generation_prompt=True)
        return f"<s>[INST] {self.STRICT_FRENCH_INSTRUCTION}\n\nQuestion: {question_fr} [/INST]"

    @torch.no_grad()
    def describe_frames(self, frames: List[Image.Image], question_fr: str, max_new_tokens: int = 96) -> str:
        prompt = self._build_prompt(question_fr)
        try:
            inputs = self.processor(text=prompt, videos=frames, return_tensors="pt")
        except Exception:
            inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")

        # Pixel values can be under different keys; avoid boolean 'or' with tensors
        pix = inputs.get("pixel_values_videos", None)
        if pix is None:
            pix = inputs.get("pixel_values", None)
        if isinstance(pix, list):
            pix = torch.stack(pix, dim=0)

        input_ids = inputs.get("input_ids", None)
        if (input_ids is None) or (hasattr(input_ids, "nelement") and input_ids.nelement() == 0):
            toks = self.tokenizer(prompt, return_tensors="pt")
            input_ids = toks["input_ids"]

        input_ids = input_ids.to(self.model.device)
        pix = pix.to(self.model.device, dtype=self.model.dtype)

        out = self.model.generate(
            input_ids=input_ids,
            pixel_values_videos=pix,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        for m in ("Assistant:", "assistant:", "ASSISTANT:"):
            if m in text:
                text = text.split(m, 1)[-1].strip()
        return text

    @torch.no_grad()
    def describe_image(self, img: Image.Image, question_fr: str) -> str:
        frames = [img.convert("RGB")] * 6
        return self.describe_frames(frames, question_fr)

print(f"[main]  PATH_LLAVA_VIDEO = {os.environ.get('PATH_LLAVA_VIDEO', '/data/models/llava_next_video')}")
VISION = _LLaVAVideo(os.environ.get("PATH_LLAVA_VIDEO", "/data/models/llava_next_video"))

print(f"[main] Models ready.")

# ---- small helpers -----------------------------------------------------------
async def _safe_close(ws: WebSocket):
    try:
        if getattr(ws, "client_state", None) != WebSocketState.DISCONNECTED:
            await ws.close()
    except Exception:
        pass

def _looks_basaa(text: str) -> bool:
    t = text.lower()
    return any(c in "ŋɓƁàáâèéêìíîòóôùúûɛɔ" for c in t)

def _normalize_tts_text(text: str) -> str:
    kept = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("L", "M", "N"):
            kept.append(ch); continue
        if ch in " .,!?:;'\"-()[]/\\\n\r\t":
            kept.append(ch)
    s = "".join(kept)
    s = re.sub(r"\s+", " ", s).strip()
    if s and s[-1] not in ".!?": s += "."
    return s

def _split_for_tts(text: str, max_len: int = 240) -> list[str]:
    if not text: return []
    pieces = re.split(r"(?<=[\.\!\?])\s+", text)
    out: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p: continue
        if len(p) <= max_len:
            out.append(p); continue
        cur: list[str] = []
        for tok in re.split(r"(\s+)", p):
            if sum(len(x) for x in cur) + len(tok) > max_len and cur:
                out.append("".join(cur).strip()); cur = [tok]
            else:
                cur.append(tok)
        if cur: out.append("".join(cur).strip())
    return out

def _is_bad_wave(w: np.ndarray) -> bool:
    if w is None or not isinstance(w, np.ndarray) or w.size == 0:
        return True
    if not np.all(np.isfinite(w)):
        return True
    if w.shape[0] < int(0.10 * TTS.sr_out):  # <100ms, use actual Orpheus SR
        return True
    if float(np.std(w)) < 1e-5:
        return True
    return False

def synthesize_wav_safe(text: str) -> np.ndarray:
    # Keep this utility for endpoints that want punctuation cleanup,
    # but we will NOT use it in /ws/audio_chat anymore.
    clean = _normalize_tts_text(text)
    chunks = _split_for_tts(clean)
    if not chunks:
        return np.zeros(int(0.3 * TTS.sr_out), dtype=np.float32)

    waves: list[np.ndarray] = []
    for chunk in chunks:
        wav = None
        try:
            wav = TTS.tts(chunk)
        except Exception:
            wav = None

        if _is_bad_wave(wav):
            simple = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]", "", chunk)
            simple = re.sub(r"\s+", " ", simple).strip()
            try:
                wav = TTS.tts(simple)
            except Exception:
                wav = None

        if _is_bad_wave(wav):
            wav = np.zeros(int(0.2 * TTS.sr_out), dtype=np.float32)

        waves.append(wav.astype(np.float32, copy=False))

    out = np.concatenate(waves, dtype=np.float32) if waves else np.zeros(0, dtype=np.float32)
    if out.size:
        peak = float(np.max(np.abs(out)))
        if peak > 0.99:
            out = out * (0.99 / peak)
    return out

# --------------------------------------------------------------------------
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"

# -------------------- text endpoints --------------------------------------
@app.websocket("/ws/translate_text")
async def ws_translate_text(ws: WebSocket):
    await ws.accept()
    try:
        payload_raw = (await ws.receive_text()).strip()
        try:
            payload = json.loads(payload_raw) if payload_raw.startswith("{") else {"text": payload_raw}
        except Exception:
            payload = {"text": payload_raw}

        text = (payload.get("text") or "").strip()
        src_hint = (payload.get("lang") or payload.get("src") or "").strip().lower()

        src = MT.resolve_safe_src(text, src_hint or None)
        out_fr = MT.to_fr(text, src)
        out_lg = MT.to_lg(text, src)

        await ws.send_text(json.dumps({"fr": out_fr, "lg": out_lg}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

@app.websocket("/ws/chat_text")
async def ws_chat_text(ws: WebSocket):
    await ws.accept()
    try:
        payload_raw = (await ws.receive_text()).strip()
        try:
            payload = json.loads(payload_raw) if payload_raw.startswith("{") else {"text": payload_raw}
        except Exception:
            payload = {"text": payload_raw}

        user_text = (payload.get("text") or "").strip()
        src_hint  = (payload.get("lang") or payload.get("src") or "").strip().lower()

        src     = MT.resolve_safe_src(user_text, src_hint or None)
        user_fr = user_text if src == "fr" else MT.to_fr(user_text, src)
        qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")

        await ws.send_text(json.dumps({"fr": qwen_fr, "lg": basaa}, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- TTS once (non-streaming WAV) ------------------------
@app.websocket("/ws/tts_once")
async def ws_tts_once(ws: WebSocket):
    await ws.accept()
    try:
        text = (await ws.receive_text()).strip()
        # Keep safety/cleanup for this endpoint if you want; main focus is audio_chat path.
        wav = synthesize_wav_safe(text)
        # IMPORTANT: write WAV with the Orpheus SR to avoid resample/pitch issues
        wav_bytes = wav_bytes_from_float32(wav, TTS.sr_out)
        try:
            await ws.send_bytes(wav_bytes)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# ---- helpers for audio sockets: read JSON header then bytes ---------------
async def _read_audio_and_header(ws: WebSocket):
    buf = bytearray()
    lang_hint = None

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg:
                    txt = (msg.get("text") or "").strip()
                    if not txt:
                        continue
                    if txt.upper() == "DONE":
                        break
                    try:
                        obj = json.loads(txt)
                        if isinstance(obj, dict):
                            lang_hint = (obj.get("lang") or obj.get("src") or lang_hint)
                    except Exception:
                        pass
                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()

    return bytes(buf), (lang_hint or "").lower().strip() or None

# -------------------- non-streaming audio: translate -----------------------
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    audio, lang_hint = await _read_audio_and_header(ws)

    try:
        text, wlang, _ = ASR_MODEL.transcribe(audio, lang_hint=lang_hint)
        src = MT.resolve_safe_src(text, lang_hint or wlang)

        fr_text = MT.to_fr(text, src)
        lg_text = MT.to_lg(text, src)

        await ws.send_text(json.dumps(
            {"asr": {"text": text, "lang": src}, "fr": fr_text, "lg": lg_text},
            ensure_ascii=False
        ))

        wav = synthesize_wav_safe(lg_text)
        # Use Orpheus SR
        wav_bytes = wav_bytes_from_float32(wav, TTS.sr_out)
        try:
            await ws.send_bytes(wav_bytes)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- non-streaming audio: chat ----------------------------
@app.websocket("/ws/audio_chat")
async def ws_audio_chat(ws: WebSocket):
    """
    RESTORED: bit-for-bit Orpheus output for the chat reply.
    - No text cleanup/normalization here.
    - WAV header uses TTS.sr_out to avoid any resample/pitch drift.
    """
    await ws.accept()
    audio, lang_hint = await _read_audio_and_header(ws)

    try:
        user_text, wlang, _ = ASR_MODEL.transcribe(audio, lang_hint=lang_hint)
        src     = MT.resolve_safe_src(user_text, lang_hint or wlang)
        user_fr = user_text if src == "fr" else MT.to_fr(user_text, src)

        qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)
        basaa   = MT.to_lg(qwen_fr, "fr")

        await ws.send_text(json.dumps(
            {"asr": {"text": user_text, "lang": src}, "fr": qwen_fr, "lg": basaa},
            ensure_ascii=False
        ))

        # CRITICAL: call Orpheus directly without normalization/splitting.
        wav = TTS.tts(basaa)
        wav_bytes = wav_bytes_from_float32(wav, TTS.sr_out)
        try:
            await ws.send_bytes(wav_bytes)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- VISION helpers & endpoint ----------------------------
def _extract_frames_from_video_bytes(blob: bytes, max_frames: int = 12) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with av.open(io.BytesIO(blob)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        if not vstreams:
            return frames
        stream = vstreams[0]
        total = max(1, int(stream.frames or 60))
        step = max(1, total // max_frames)
        idx = 0
        for packet in container.demux(stream):
            for f in packet.decode():
                if idx % step == 0 and len(frames) < max_frames:
                    img = f.to_image().convert("RGB")
                    frames.append(img)
                idx += 1
                if len(frames) >= max_frames:
                    break
            if len(frames) >= max_frames:
                break
    return frames

@app.websocket("/ws/vision_once")
async def ws_vision_once(ws: WebSocket):
    """
    Contract:
      - (optional) first TEXT frame: {"question":"Qu’est-ce que tu vois ?"}
      - one BINARY frame: image (PNG/JPEG) **or** short video (mp4/webm)
      - then client sends "DONE"
    Server replies:
      - TEXT JSON once: {"fr":"...", "lg":"..."}
      - BINARY once: full WAV (Basaa TTS of 'lg')
    """
    await ws.accept()
    question = "Décris brièvement ce que tu vois."
    blob = bytearray()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if msg.get("text"):
                    txt = msg["text"].strip()
                    if txt.upper() == "DONE":
                        break
                    try:
                        obj = json.loads(txt)
                        if isinstance(obj, dict) and "question" in obj:
                            q = str(obj["question"]).strip()
                            if q: question = q
                    except Exception:
                        pass
                elif msg.get("bytes"):
                    blob.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                return

        if not blob:
            await ws.send_text(json.dumps({"error": "no image/video payload"}, ensure_ascii=False))
            return

        data = bytes(blob)
        frames: list[Image.Image] = []
        # Try image first
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            frames = [img] * 6
        except Exception:
            # Not an image → try video
            try:
                frames = _extract_frames_from_video_bytes(data, max_frames=12)
            except Exception:
                frames = []

        if not frames:
            await ws.send_text(json.dumps({"error": "could not decode image/video"}, ensure_ascii=False))
            return

        try:
            fr = VISION.describe_frames(frames, question_fr=question)
        except Exception as e:
            await ws.send_text(json.dumps({"error": f"vision_failed: {type(e).__name__}: {e}"}, ensure_ascii=False))
            return

        lg = MT.to_lg(fr, "fr")

        await ws.send_text(json.dumps({"fr": fr, "lg": lg}, ensure_ascii=False))

        wav = synthesize_wav_safe(lg)
        await ws.send_bytes(wav_bytes_from_float32(wav, TTS.sr_out))

    except Exception:
        traceback.print_exc()
    finally:
        await _safe_close(ws)

# -------------------- aliases kept for client compatibility ----------------
@app.websocket("/ws/translate_stream")
async def ws_translate_stream(ws: WebSocket):
    await ws_translate(ws)

@app.websocket("/ws/audio_chat_stream")
async def ws_audio_chat_stream(ws: WebSocket):
    await ws_audio_chat(ws)
