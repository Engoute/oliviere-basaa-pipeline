from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import os, numpy as np, torch

# Force no torchvision import
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from transformers import AutoModelForSpeechSeq2Seq

__all__ = ["WhisperGeneralASR", "WhisperBasaaASR"]

# ---------------- fallbacks for audio decode (used by wrappers) ----------------
try:
    from .utils_audio import decode_audio_to_16k_float_mono  # type: ignore
except Exception:
    def decode_audio_to_16k_float_mono(payload: bytes) -> np.ndarray:  # type: ignore
        try:
            import io, torchaudio
            buf = io.BytesIO(payload)
            wav, sr = torchaudio.load(buf)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            try:
                import io, wave
                buf = io.BytesIO(payload)
                with wave.open(buf, "rb") as w:
                    nch = w.getnchannels(); sr = w.getframerate(); n = w.getnframes()
                    sw  = w.getsampwidth(); raw = w.readframes(n)
                if sw == 2:
                    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                elif sw == 1:
                    x = (np.frombuffer(raw, dtype="|u1").astype(np.float32) - 128.0) / 128.0
                else:
                    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                if nch > 1:
                    x = x.reshape(-1, nch).mean(axis=1)
                return x.astype(np.float32, copy=False)
            except Exception:
                x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                return x

BASAA_ALIASES = {"lg", "bas", "basaa"}

def _is_hf_dir(d: Path) -> bool:
    return (d / "config.json").exists() and (
        (d / "processor").exists()
        or (d / "tokenizer.json").exists()
        or (d / "tokenizer_config.json").exists()
        or (d / "feature_extractor.json").exists()
        or (d / "preprocessor_config.json").exists()
    )

def _resolve_hf_dir(base: Path) -> Optional[Path]:
    if _is_hf_dir(base): return base
    best, score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        s = (2 if (d / "preprocessor_config.json").exists() else 0) \
            + (2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0) \
            + (1 if (d / "feature_extractor.json").exists() or (d / "processor").exists() else 0)
        if s > score:
            best, score = d, s
    return best

def _safe_max_new_tokens(model) -> int:
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            mtp = int(getattr(cfg, "max_target_positions", 448) or 448)
            return max(32, min(400, mtp - 8))
    except Exception:
        pass
    return 256

def _build_processor(use: Path, prefer_auto: bool):
    def _classic():
        from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
        tok = None; fe = None
        try:
            tok = WhisperTokenizer.from_pretrained(str(use), subfolder="processor", local_files_only=True, use_fast=False)
        except Exception:
            tok = WhisperTokenizer.from_pretrained(str(use), local_files_only=True, use_fast=False)
        try:
            fe = WhisperFeatureExtractor.from_pretrained(str(use), subfolder="processor", local_files_only=True)
        except Exception:
            fe = WhisperFeatureExtractor.from_pretrained(str(use), local_files_only=True)
        return WhisperProcessor(feature_extractor=fe, tokenizer=tok)

    if not prefer_auto:
        return _classic()

    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(str(use), local_files_only=True, trust_remote_code=True)
    except Exception:
        return _classic()

class _Core:
    def __init__(self, root: Path, device: str, dtype: torch.dtype, prefer_auto: bool):
        use = _resolve_hf_dir(root) or root
        self.proc  = _build_processor(use, prefer_auto=prefer_auto)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(use),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        self.device = device

        self.lang_to_id = {}
        tok = getattr(self.proc, "tokenizer", None)
        if tok is not None:
            for code in (
                "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha he hi hr ht hu hy id "
                "is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so "
                "sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue lg bas basaa"
            ).split():
                tid = tok.convert_tokens_to_ids(f"<|{code}|>")
                if isinstance(tid, int) and tid >= 0:
                    self.lang_to_id[code] = tid

        self._tok  = tok
        self._sot  = tok.convert_tokens_to_ids("<|startoftranscript|>") if tok else None
        self._not  = tok.convert_tokens_to_ids("<|notimestamps|>") if tok else None
        self._task = tok.convert_tokens_to_ids("<|transcribe|>") if tok else None

    def transcribe(self, wav16k: np.ndarray, lang_hint: Optional[str] = None) -> Tuple[str, str, float]:
        inputs = self.proc(wav16k, sampling_rate=16000, return_tensors="pt")
        feats = inputs["input_features"].to(self.model.device, dtype=self.model.dtype)

        forced_ids = None
        if self._tok is not None and None not in (self._sot, self._task, self._not):
            lang_tok = None
            if lang_hint:
                hint = lang_hint.lower()
                if hint in BASAA_ALIASES: hint = "lg"
                if hint in self.lang_to_id:
                    lang_tok = self.lang_to_id[hint]
            # STRICT: if we have a hint and it's valid -> we do NOT fallback to detected language
            if lang_tok is not None:
                forced_ids = [(0, self._sot), (1, lang_tok), (2, self._task), (3, self._not)]
            else:
                # no hint provided -> allow internal detection, but still fix <|transcribe|>,<|notimestamps|>
                forced_ids = [(0, self._sot), (1, self._task), (2, self._not)]

        try:
            self.model.generation_config.forced_decoder_ids = forced_ids
        except Exception:
            pass

        max_new = _safe_max_new_tokens(self.model)
        with torch.inference_mode():
            out = self.model.generate(
                feats,
                do_sample=False, num_beams=1,
                max_new_tokens=max_new,
                early_stopping=True,
            )
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()

        # optional detection (not used when hint is set)
        code = "unk"; conf = 0.0
        try:
            if self._sot is not None and self._not is not None and self.lang_to_id:
                dec = torch.tensor([[self._sot, self._not]], device=self.model.device, dtype=torch.long)
                with torch.inference_mode():
                    logits = self.model(input_features=feats, decoder_input_ids=dec).logits[:, -1, :]
                    probs  = torch.softmax(logits, dim=-1)[0]
                code, tid = max(self.lang_to_id.items(), key=lambda kv: float(probs[kv[1]].item()))
                conf = float(probs[tid].item())
        except Exception:
            pass

        if code.lower() in BASAA_ALIASES: code = "lg"
        return text, code, conf

# ---------------- thin wrappers to preserve imports used by api_ws.py ---------------

class WhisperGeneralASR:
    def __init__(self, local_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        root = Path(local_dir or "/data/models/whisper_general_resolved")
        self.core = _Core(root, self.device, self.dtype, prefer_auto=True)

    def transcribe(self, x16: np.ndarray, lang: Optional[str] = None) -> str:
        # STRICT: pass the user-selected hint ("fr" or "en") to force language tokens.
        text, _, _ = self.core.transcribe(x16, lang_hint=(lang or "").lower().strip() or None)
        return text

class WhisperBasaaASR:
    def __init__(self, local_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        root = Path(local_dir or "/data/models/whisper_hf_resolved")
        # Basaa core ALWAYS uses classic processor (prefer_auto=False)
        self.core = _Core(root, self.device, self.dtype, prefer_auto=False)

    def transcribe(self, x16: np.ndarray, lang: Optional[str] = None) -> str:
        # For Basaa model we ignore lang and force LG behavior.
        text, _, _ = self.core.transcribe(x16, lang_hint="lg")
        return text
