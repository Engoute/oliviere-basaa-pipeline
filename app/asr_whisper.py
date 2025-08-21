from pathlib import Path
import numpy as np, torch
from typing import Optional

from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASAA_ALIASES = {"lg", "bas", "basaa"}  # normalize to "lg"

def _resolve_hf_root(base: Path) -> Optional[Path]:
    """
    Find a folder that has a Whisper HF merge:
      - config.json
      - and either tokenizer.json/tokenizer_config.json or a processor/ subfolder.
    """
    def _is_good(d: Path) -> bool:
        return (d / "config.json").exists() and (
            (d / "tokenizer.json").exists()
            or (d / "tokenizer_config.json").exists()
            or (d / "processor").exists()
        )

    if _is_good(base): 
        return base
    # search nested
    best = None
    best_score = -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        score = (2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0) \
              + (1 if (d / "processor").exists() else 0)
        if score > best_score:
            best, best_score = d, score
    return best

class ASR:
    def __init__(self, base_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        # Prefer bootstrap symlink; then the provided path
        candidates = [Path("/data/models/whisper_hf_resolved")]
        if base_path:
            candidates.insert(0, Path(base_path))  # let explicit path win

        root = None
        for c in candidates:
            if c.exists():
                root = _resolve_hf_root(c)
                if root is not None:
                    break
        if root is None:
            raise RuntimeError(f"Whisper HF bundle not found under {candidates}")

        # EXACTLY like your Colab test:
        self.proc  = WhisperProcessor.from_pretrained(str(root), subfolder="processor", local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(root), local_files_only=True, torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()

        self.tok = self.proc.tokenizer
        self.fe  = self.proc.feature_extractor

        # Build language token map once (same set we used on Colab)
        CODES = ("af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha he hi hr ht hu hy id "
                 "is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so "
                 "sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue lg bas basaa").split()
        self.lang_to_id = {}
        for c in CODES:
            tid = self.tok.convert_tokens_to_ids(f"<|{c}|>")
            if isinstance(tid, int) and tid >= 0:
                self.lang_to_id[c] = tid

        print(f"[asr] Whisper HF ready @ {root}")

    @staticmethod
    def _pcm16_to_float(wav16k: bytes) -> np.ndarray:
        return np.frombuffer(wav16k, dtype=np.int16).astype(np.float32) / 32768.0

    def _detect_lang(self, feats):
        # One decoder step like in Colab
        sot  = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        nots = self.tok.convert_tokens_to_ids("<|notimestamps|>")
        dec  = torch.tensor([[sot, nots]], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out   = self.model(input_features=feats, decoder_input_ids=dec)
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        code, tid = max(self.lang_to_id.items(), key=lambda kv: float(probs[kv[1]].item())) if self.lang_to_id else ("unk", None)
        p = float(probs[tid].item()) if tid is not None else 0.0
        return code, p

    def transcribe(self, wav16k: bytes):
        pcm   = self._pcm16_to_float(wav16k)
        feats = self.proc(audio=pcm, sampling_rate=16000, return_tensors="pt").input_features.to(self.device).to(self.model.dtype)

        code, conf = self._detect_lang(feats)

        # Generate EXACTLY like Colab: pass kwargs; DO NOT touch generation_config objects.
        max_target = int(getattr(self.model.config, "max_target_positions", 448) or 448)
        safe_new   = max(1, min(400, max_target - 3))
        with torch.inference_mode():
            out = self.model.generate(
                feats,
                do_sample=False, num_beams=1,
                max_new_tokens=safe_new,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()

        code = "lg" if (code or "unk").lower() in BASAA_ALIASES else (code or "unk").lower()
        return text, code, conf
