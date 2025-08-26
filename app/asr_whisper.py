# FILE: app/asr_whisper.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import os

# 🚫 Prevent transformers from importing torchvision (fixes "operator torchvision::nms does not exist")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Try to use the project's decoder; fall back to a local implementation if absent.
try:
    from .utils_audio import decode_audio_to_16k_float_mono  # type: ignore
except Exception:
    # Minimal fallback decoder: torchaudio / soundfile / raw PCM16
    def decode_audio_to_16k_float_mono(payload: bytes) -> np.ndarray:  # type: ignore
        try:
            import io, torchaudio  # noqa: F401
            buf = io.BytesIO(payload)
            wav, sr = torchaudio.load(buf)  # [C,T]
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            try:
                import io, soundfile as sf  # type: ignore
                buf = io.BytesIO(payload)
                audio, sr = sf.read(buf, dtype="float32", always_2d=True)
                if sr != 16000:
                    import librosa  # type: ignore
                    audio = librosa.resample(audio.T, orig_sr=sr, target_sr=16000).T
                mono = audio.mean(axis=1).astype(np.float32, copy=False)
                return mono
            except Exception:
                x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
                return x

__all__ = ["ASR"]

BASAA_ALIASES = {"lg", "bas", "basaa"}

def _is_hf_dir(d: Path) -> bool:
    return (d / "config.json").exists() and (
        (d / "processor").exists()
        or (d / "tokenizer.json").exists()
        or (d / "tokenizer_config.json").exists()
    )

def _resolve_hf_dir(base: Path) -> Optional[Path]:
    if _is_hf_dir(base):
        return base
    best, score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        s = (2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0) \
            + (1 if (d / "processor").exists() else 0)
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

class _Core:
    def __init__(self, root: Path, device: str, dtype: torch.dtype):
        use = _resolve_hf_dir(root) or root
        self.proc  = AutoProcessor.from_pretrained(str(use), local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(use),
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        self.device = device

        # Build language token id map if tokenizer exposes them
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

    def _detect_lang(self, feats) -> Tuple[str, float]:
        if not self.lang_to_id or self._sot is None or self._not is None:
            return "unk", 0.0
        dec = torch.tensor([[self._sot, self._not]], device=self.model.device, dtype=torch.long)
        with torch.inference_mode():
            out   = self.model(input_features=feats, decoder_input_ids=dec)
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        code, tid = max(self.lang_to_id.items(), key=lambda kv: float(probs[kv[1]].item()))
        return code, float(probs[tid].item())

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
            if lang_tok is None:
                code, _ = self._detect_lang(feats)
                if code in self.lang_to_id:
                    lang_tok = self.lang_to_id[code]
            if lang_tok is not None:
                forced_ids = [(0, self._sot), (1, lang_tok), (2, self._task), (3, self._not)]
            else:
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

        code, conf = self._detect_lang(feats)
        code = "lg" if (code or "unk").lower() in BASAA_ALIASES else (code or "unk").lower()
        return text, code, conf

class ASR:
    """
    Dual ASR router:
      - Basaa-finetuned Whisper for 'lg'
      - Whisper v3 (general) for fr/en/others
    """
    def __init__(self, path_basaa: Optional[str] = None, path_general: Optional[str] = None, role: Optional[str] = None):
        print("[asr] importing app.asr_whisper (dual-capable)…")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        self.core_basaa: Optional[_Core] = None
        self.core_general: Optional[_Core] = None

        exp_basaa   = [p for p in [path_basaa, "/data/models/whisper_basaa_resolved", "/data/models/whisper_hf_resolved"] if p]
        exp_general = [p for p in [path_general, "/data/models/whisper_general_resolved"] if p]

        def _try_load(where: str) -> Optional[_Core]:
            p = Path(where)
            if not p.exists():
                return None
            try:
                return _Core(p, self.device, self.dtype)
            except Exception as e:
                print(f"[asr] WARN: could not load core at {where}: {e}")
                return None

        if role in ("basaa", "general") and path_basaa and not path_general:
            core = _try_load(path_basaa)
            if role == "basaa":
                self.core_basaa = core
            else:
                self.core_general = core

        if self.core_basaa is None:
            for p in exp_basaa:
                core = _try_load(p)
                if core:
                    self.core_basaa = core
                    print(f"[asr] Basaa core loaded from: {p}")
                    break

        if self.core_general is None:
            for p in exp_general:
                core = _try_load(p)
                if core:
                    self.core_general = core
                    print(f"[asr] General core loaded from: {p}")
                    break

        if not self.core_basaa and not self.core_general and path_basaa and not path_general and role is None:
            core = _try_load(path_basaa)
            self.core_general = core
            print(f"[asr] Single core loaded (generic): {path_basaa}")

        if not self.core_basaa and not self.core_general:
            raise RuntimeError("No ASR cores available. Check Whisper bundles on disk.")

    def transcribe(self, audio_bytes: bytes, lang_hint: Optional[str] = None) -> Tuple[str, str, float]:
        wav16 = decode_audio_to_16k_float_mono(audio_bytes)
        hint = (lang_hint or "").lower().strip()
        use_basaa = (hint in BASAA_ALIASES) and (self.core_basaa is not None)
        core = self.core_basaa if use_basaa else (self.core_general or self.core_basaa)
        text, code, conf = core.transcribe(wav16, lang_hint=hint or None)
        return text, code, conf
