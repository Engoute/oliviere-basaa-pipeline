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
import time
from typing import List, Optional, Tuple
from PIL import Image, ImageOps, ImageEnhance
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
        "Tu es un assistant de vision. Réponds en français en 1 à 2 phrases courtes, "
        "sans métadonnées ni balises. Décris uniquement ce qui est VISIBLE, de façon précise. "
        "N'invente pas de détails. Si l'image est entièrement noire/blanche, très floue "
        "ou ne montre rien de discernable, tu peux répondre : «Je ne suis pas sûr.»"
    )
    SOFT_FRENCH_INSTRUCTION = (
        "Décris brièvement et précisément ce que tu vois sur l'image. "
        "Réponds en français en 1 à 2 phrases naturelles."
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

    def _build_prompt(self, question_fr: str, strict: bool = True) -> str:
        instr = self.STRICT_FRENCH_INSTRUCTION if strict else self.SOFT_FRENCH_INSTRUCTION
        conv = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": f"{instr}\n\nQuestion: {question_fr}"},
            ],
        }]
        apply = getattr(self.tokenizer, "apply_chat_template", None) or getattr(self.processor, "apply_chat_template", None)
        if apply:
            try:
                return apply(conv, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return apply(conv, add_generation_prompt=True)
        return f"<s>[INST] {instr}\n\nQuestion: {question_fr} [/INST]"

    def _prep_inputs(self, frames: List[Image.Image], prompt: str):
        # Try “videos=frames”; some processors also accept videos=[frames]
        try:
            inputs = self.processor(text=prompt, videos=frames, return_tensors="pt")
        except Exception:
            inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")

        # Determine which pixel key exists
        pixel_key = None
        if "pixel_values_videos" in inputs:
            pixel_key = "pixel_values_videos"
            pix = inputs["pixel_values_videos"]
        elif "pixel_values" in inputs:
            pixel_key = "pixel_values"
            pix = inputs["pixel_values"]
        else:
            raise RuntimeError("Processor did not return pixel values")

        if isinstance(pix, list):
            pix = torch.stack(pix, dim=0)

        input_ids = inputs.get("input_ids", None)
        if (input_ids is None) or (hasattr(input_ids, "nelement") and input_ids.nelement() == 0):
            toks = self.tokenizer(prompt, return_tensors="pt")
            input_ids = toks["input_ids"]

        input_ids = input_ids.to(self.model.device)
        pix = pix.to(self.model.device, dtype=self.model.dtype)
        return input_ids, pix, pixel_key

    @torch.no_grad()
    def _generate(self, input_ids, pix, pixel_key: str, max_new_tokens: int, sample: bool):
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # attach pixels under the exact key name the model expects
        gen_kwargs[pixel_key] = pix

        if sample:
            gen_kwargs.update(dict(do_sample=True, temperature=0.2, top_p=0.9))
        else:
            # Deterministic path: do NOT set temperature/top_p to avoid HF warning.
            gen_kwargs.update(dict(do_sample=False))

        out = self.model.generate(**gen_kwargs)
        return out

    def _post(self, out, prompt: str) -> str:
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        for m in ("Assistant:", "assistant:", "ASSISTANT:"):
            if m in text:
                text = text.split(m, 1)[-1].strip()
        return text

    @torch.no_grad()
    def describe_frames(self, frames: List[Image.Image], question_fr: str, max_new_tokens: int = 96) -> str:
        # Pass 1: strict, deterministic
        prompt = self._build_prompt(question_fr, strict=True)
        input_ids, pix, pixel_key = self._prep_inputs(frames, prompt)
        out = self._generate(input_ids, pix, pixel_key, max_new_tokens=max_new_tokens, sample=False)
        text = self._post(out, prompt)

        low = text.lower()
        if "je ne suis pas sûr" in low or "je ne suis pas sur" in low:
            # Pass 2: softer, allow tiny sampling to escape the safety basin
            prompt2 = self._build_prompt(question_fr, strict=False)
            input_ids2, pix2, pixel_key2 = self._prep_inputs(frames, prompt2)
            out2 = self._generate(input_ids2, pix2, pixel_key2, max_new_tokens=max_new_tokens, sample=True)
            text2 = self._post(out2, prompt2)
            if text2.strip():
                text = text2
        return text

    @torch.no_grad()
    def describe_image(self, img: Image.Image, question_fr: str) -> str:
        # treat image as a short “video” (repeat few frames)
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
    # no extra trim; Orpheus already adds a tiny tail pad
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
        # honor nested ASR language pins too
        asr_obj = payload.get("asr") if isinstance(payload, dict) else None
        src_hint = (payload.get("lang") or payload.get("src") or
                    (isinstance(asr_obj, dict) and (asr_obj.get("language") or asr_obj.get("lang") or asr_obj.get("src"))) or
                    "").strip().lower()

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
        asr_obj  = payload.get("asr") if isinstance(payload, dict) else None
        src_hint  = (payload.get("lang") or payload.get("src") or
                     (isinstance(asr_obj, dict) and (asr_obj.get("language") or asr_obj.get("lang") or asr_obj.get("src"))) or
                     "").strip().lower()

        src     = MT.resolve_safe_src(user_text, src_hint or None)
        user_fr = user_text if src == "fr" else MT.to_fr(user_text, src)

        # Tiny heuristic: if it's just a greeting, answer naturally instead of "how can I help?"
        if re.search(r"\b(bonjour|salut|bonsoir)\b", user_fr, flags=re.I) and re.search(r"\b(comment\s*(ça|ca)\s*va|tu vas)\b", user_fr, flags=re.I):
            qwen_fr = "Je vais bien, merci ! Et toi, ça va ?"
        else:
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
        wav = synthesize_wav_safe(text)
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
async def _read_audio_and_header(ws: WebSocket) -> Tuple[bytes, Optional[str], Optional[str]]:
    """
    Collects audio bytes + optional {'lang'| 'src'} or nested {'asr': {'language'|'lang'|'src'}} +
    optional {'text'} or {'asr': {'text'}} override.
    Ends when the client sends "DONE" (or {"type":"stop"}).
    Returns: (audio_bytes, lang_hint, text_override)
    """
    buf = bytearray()
    lang_hint: Optional[str] = None
    typed_text: Optional[str] = None

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive":
                if "text" in msg and msg["text"] is not None:
                    txt = (msg.get("text") or "").strip()
                    if not txt:
                        continue
                    # stop?
                    try:
                        obj = json.loads(txt)
                        if isinstance(obj, dict) and str(obj.get("type","")).lower() == "stop":
                            break
                    except Exception:
                        pass
                    if txt.upper() == "DONE":
                        break
                    # parse fields
                    try:
                        obj = json.loads(txt)
                        if isinstance(obj, dict):
                            if obj.get("lang"):
                                lang_hint = (obj.get("lang") or obj.get("src") or lang_hint)
                            if obj.get("src"):
                                lang_hint = (obj.get("src") or lang_hint)
                            # nested ASR
                            asr = obj.get("asr")
                            if isinstance(asr, dict):
                                if asr.get("language"):
                                    lang_hint = asr.get("language")
                                elif asr.get("lang"):
                                    lang_hint = asr.get("lang")
                                elif asr.get("src"):
                                    lang_hint = asr.get("src")
                                if isinstance(asr.get("text"), str) and asr["text"].strip():
                                    typed_text = asr["text"].strip()
                            if isinstance(obj.get("text"), str) and obj["text"].strip():
                                typed_text = obj["text"].strip()
                    except Exception:
                        # ignore non-JSON status lines
                        pass
                elif "bytes" in msg and msg["bytes"]:
                    buf.extend(msg["bytes"])
            elif msg["type"] == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()

    lang_hint = (lang_hint or "").lower().strip() or None
    return bytes(buf), lang_hint, typed_text

# -------------------- non-streaming audio: translate -----------------------
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    await ws.accept()
    audio, lang_hint, typed = await _read_audio_and_header(ws)

    try:
        if typed and typed.strip():
            text = typed.strip()
            src = MT.resolve_safe_src(text, lang_hint or None)
        else:
            text, wlang, _ = ASR_MODEL.transcribe(audio, lang_hint=lang_hint)
            src = MT.resolve_safe_src(text, lang_hint or wlang)

        fr_text = MT.to_fr(text, src)
        lg_text = MT.to_lg(text, src)

        await ws.send_text(json.dumps(
            {"asr": {"text": text, "lang": src}, "fr": fr_text, "lg": lg_text},
            ensure_ascii=False
        ))

        wav = synthesize_wav_safe(lg_text)
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
    RESTORED + EXTENDED:
      - Honors a text override: client can send {"text":"..."} then "DONE".
      - If no text, falls back to ASR on the audio payload.
      - WAV header uses TTS.sr_out to avoid resample/pitch drift.
    """
    await ws.accept()
    audio, lang_hint, typed = await _read_audio_and_header(ws)

    try:
        if typed and typed.strip():
            user_text = typed.strip()
            src = MT.resolve_safe_src(user_text, lang_hint or None)
            user_fr = user_text if src == "fr" else MT.to_fr(user_text, src)
        else:
            user_text, wlang, _ = ASR_MODEL.transcribe(audio, lang_hint=lang_hint)
            src     = MT.resolve_safe_src(user_text, lang_hint or wlang)
            user_fr = user_text if src == "fr" else MT.to_fr(user_text, src)

        # Greeting micro-heuristic (less “comment puis-je aider ?” on greetings)
        if re.search(r"\b(bonjour|salut|bonsoir)\b", user_fr, flags=re.I) and re.search(r"\b(comment\s*(ça|ca)\s*va|tu vas)\b", user_fr, flags=re.I):
            qwen_fr = "Je vais bien, merci ! Et toi, ça va ?"
        else:
            qwen_fr = QWEN.chat_fr(user_fr, temperature=0.2)

        basaa   = MT.to_lg(qwen_fr, "fr")

        await ws.send_text(json.dumps(
            {"asr": {"text": user_text, "lang": src}, "fr": qwen_fr, "lg": basaa},
            ensure_ascii=False
        ))

        # Orpheus direct (no text normalization here)
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

# -------------------- VISION debug helpers ---------------------------------
def _save_debug_blob(data: bytes, ext_hint: str = "bin") -> str:
    ts = int(time.time() * 1000)
    path = f"/tmp/vision_in_{ts}.{ext_hint}"
    try:
        with open(path, "wb") as f:
            f.write(data)
        print(f"[vision] saved payload -> {path} ({len(data)} bytes)")
    except Exception as e:
        print(f"[vision] failed to save payload: {e}")
    return path

def _mean_brightness(img: Image.Image) -> float:
    # 0..255 mean over all channels
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype="float32")
    return float(arr.mean())

def _brighten_if_too_dark(img: Image.Image, threshold: float = 22.0) -> tuple[Image.Image, float, bool]:
    """
    If image mean brightness is below threshold, apply autocontrast + brightness boost.
    Returns (possibly-updated-image, mean_after, did_modify)
    """
    m0 = _mean_brightness(img)
    if m0 >= threshold:
        return img, m0, False
    try:
        # Autocontrast then brighten ~2x; tune conservatively
        img2 = ImageOps.autocontrast(img, cutoff=1)
        img2 = ImageEnhance.Brightness(img2).enhance(2.0)
        m1 = _mean_brightness(img2)
        print(f"[vision] brighten: mean {m0:.1f} -> {m1:.1f}")
        return img2, m1, True
    except Exception as e:
        print(f"[vision] brighten failed: {e}")
        return img, m0, False

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

        # --- save the raw payload so we can inspect what the client really sent ---
        ext = "bin"
        if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            ext = "png"
        elif len(data) >= 2 and data[:2] == b"\xFF\xD8":
            ext = "jpg"
        elif data[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00 ", b"RIFF"):
            ext = "mp4"
        _save_debug_blob(data, ext)

        # Decode to frames
        frames: list[Image.Image] = []
        # Try image first
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            print(f"[vision] image decoded: {img.size}, mean={_mean_brightness(img):.1f}")
            img, _, did = _brighten_if_too_dark(img)
            if did:
                ts = int(time.time() * 1000)
                bright_path = f"/tmp/vision_bright_{ts}.jpg"
                try:
                    img.save(bright_path, "JPEG", quality=92)
                    print(f"[vision] wrote brightened image -> {bright_path}")
                except Exception:
                    pass
            frames = [img] * 6
        except Exception:
            # Not an image → try video
            try:
                frames = _extract_frames_from_video_bytes(data, max_frames=12)
                if frames:
                    m = _mean_brightness(frames[0])
                    print(f"[vision] video frames: {len(frames)}, first={frames[0].size}, mean={m:.1f}")
                    fr0, _, did = _brighten_if_too_dark(frames[0])
                    if did:
                        frames[0] = fr0
            except Exception as e:
                print(f"[vision] video decode failed: {e}")
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
