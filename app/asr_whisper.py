from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq

# keep torchvision disabled (avoids op registration issues)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

__all__ = ["ASR", "WhisperGeneralASR", "WhisperBasaaASR"]

BASAA_ALIASES = {"lg", "bas", "basaa"}

def _is_hf_dir(d: Path) -> bool:
    return (d / "config.json").exists() and (
        (d / "processor").exists()
        or (d / "tokenizer.json").exists()
        or (d / "tokenizer_config.json").exists()
        or (d / "feature_extractor.json").exists()
        or (d / "preprocessor_config.json").exists()
    )

def _resolve_hf_dir(base: Path) -> Path:
    if _is_hf_dir(base):
        return base
    best, score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        s = (2 if (d / "preprocessor_config.json").exists() else 0) \
          + (2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0) \
          + (1 if (d / "feature_extractor.json").exists() or (d / "processor").exists() else 0)
        if s > score:
            best, score = d, s
    return best or base

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
    # Classic WhisperProcessor loader (Basaa-safe)
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

    # AutoProcessor for general bundle, fallback to classic
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(str(use), local_files_only=True, trust_remote_code=True)
    except Exception:
        return _classic()

class _Core:
    def __init__(self, root: str, prefer_auto: bool):
        use = _resolve_hf_dir(Path(root))
        self.proc  = _build_processor(use, prefer_auto=prefer_auto)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(use),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            low_cpu_mem_usage=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

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

    def transcribe_array(self, wav16k: np.ndarray, lang_hint: Optional[str] = None) -> Tuple[str, str, float]:
        """Input: float32 mono 16k array. Returns (text, lang_code, confidence)."""
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

            # IMPORTANT: force GENERAL Whisper to a single language when hint is provided (fr/en).
            # This is the official forcing route via forced_decoder_ids.
            if lang_tok is not None:
                forced_ids = [(0, self._sot), (1, lang_tok), (2, self._task), (3, self._not)]
            else:
                forced_ids = [(0, self._sot), (1, self._task), (2, self._not)]

        try:
            self.model.generation_config.forced_decoder_ids = forced_ids
        except Exception:
            pass

        with torch.inference_mode():
            out = self.model.generate(
                feats,
                do_sample=False, num_beams=1,
                max_new_tokens=_safe_max_new_tokens(self.model),
                early_stopping=True,
            )
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()

        code, conf = self._detect_lang(feats)
        code = "lg" if (code or "unk").lower() in BASAA_ALIASES else (code or "unk").lower()
        return text, code, conf

# -------- public wrappers (compatible names) --------
class WhisperGeneralASR:
    """General Whisper with language forcing when lang is supplied (fr/en)."""
    def __init__(self, path: Optional[str] = None):
        self.core = _Core(path or "/data/models/whisper_general_resolved", prefer_auto=True)
    def transcribe(self, wav16k: np.ndarray, lang: Optional[str] = None) -> str:
        text, *_ = self.core.transcribe_array(wav16k, lang_hint=lang)
        return text

class WhisperBasaaASR:
    """Basaa-finetuned Whisper (never uses AutoProcessor)."""
    def __init__(self, path: Optional[str] = None):
        self.core = _Core(path or "/data/models/whisper_hf_resolved", prefer_auto=False)
    def transcribe(self, wav16k: np.ndarray) -> str:
        text, *_ = self.core.transcribe_array(wav16k, lang_hint="lg")
        return text

# unified class kept for modules that import ASR
class ASR:
    def __init__(self, path_basaa: Optional[str] = None, path_general: Optional[str] = None):
        self.basaa = WhisperBasaaASR(path_basaa)
        self.general = WhisperGeneralASR(path_general)
    def transcribe(self, audio_or_array: Union[bytes, np.ndarray], lang_hint: Optional[str] = None) -> Tuple[str,str,float]:
        if isinstance(audio_or_array, (bytes, bytearray, memoryview)):
            from .utils_audio import decode_audio_to_16k_float_mono
            arr = decode_audio_to_16k_float_mono(bytes(audio_or_array))
        else:
            arr = np.asarray(audio_or_array, dtype=np.float32)
        core = self.general if (lang_hint or "fr") not in BASAA_ALIASES else self.basaa
        if core is self.general:
            text = core.transcribe(arr, lang=lang_hint)
            # echo hint as code for downstream
            return text, (lang_hint or "fr"), 1.0
        else:
            text = core.transcribe(arr)
            return text, "lg", 1.0
