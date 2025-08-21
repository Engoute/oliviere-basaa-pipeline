# app/asr_whisper.py
from pathlib import Path
import numpy as np
import torch

try:
    from faster_whisper import WhisperModel as CT2WhisperModel  # type: ignore
except Exception:
    CT2WhisperModel = None

try:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
        WhisperFeatureExtractor,
        GenerationConfig,  # <-- added
    )  # type: ignore
    _WHISPER_NATIVE = True
except Exception:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoTokenizer,
        AutoFeatureExtractor,
        GenerationConfig,  # still available
    )  # type: ignore
    WhisperForConditionalGeneration = AutoModelForSpeechSeq2Seq  # type: ignore
    WhisperTokenizer = AutoTokenizer  # type: ignore
    WhisperFeatureExtractor = AutoFeatureExtractor  # type: ignore
    WhisperProcessor = None  # type: ignore
    _WHISPER_NATIVE = False

BASAA_ALIASES = {"lg", "bas", "basaa"}


def _is_hf_dir(d: Path) -> bool:
    return (d / "config.json").exists() and (
        (d / "tokenizer.json").exists()
        or (d / "tokenizer_config.json").exists()
        or (d / "processor").exists()
    )


def _resolve_hf_dir(base: Path) -> Path | None:
    if _is_hf_dir(base):
        return base
    best, best_score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        score = (
            2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0
        ) + (1 if (d / "processor").exists() else 0)
        if score > best_score:
            best, best_score = d, score
    return best


class ASR:
    def __init__(self, base_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        candidates = [Path("/data/models/whisper_hf_resolved")]
        if base_path:
            candidates.append(Path(base_path))

        use_hf = None
        use_ct2 = None
        for c in candidates:
            if c.exists():
                if (c / "model.bin").exists():
                    use_ct2 = c
                    break
                d = _resolve_hf_dir(c)
                if d is not None:
                    use_hf = d
                    break

        if use_hf is not None:
            # processor / tokenizer / feature extractor
            if _WHISPER_NATIVE and WhisperProcessor is not None:
                try:
                    self.proc = WhisperProcessor.from_pretrained(
                        str(use_hf), subfolder="processor", local_files_only=True
                    )
                except Exception:
                    self.proc = WhisperProcessor.from_pretrained(str(use_hf), local_files_only=True)
                self.tok = self.proc.tokenizer
                self.feat = self.proc.feature_extractor
            else:
                self.tok = WhisperTokenizer.from_pretrained(str(use_hf), local_files_only=True)
                try:
                    self.feat = WhisperFeatureExtractor.from_pretrained(
                        str(use_hf), subfolder="processor", local_files_only=True
                    )
                except Exception:
                    self.feat = WhisperFeatureExtractor.from_pretrained(
                        str(use_hf), local_files_only=True
                    )

            self.hf = WhisperForConditionalGeneration.from_pretrained(
                str(use_hf),
                local_files_only=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()

            # ---- SAFE GENERATION CONFIG (avoids early_stopping=None) ----
            try:
                safe_gen = GenerationConfig.from_model_config(self.hf.config)
                # force sane defaults if missing
                if getattr(safe_gen, "early_stopping", None) is None:
                    safe_gen.early_stopping = False
                if getattr(safe_gen, "pad_token_id", None) is None:
                    safe_gen.pad_token_id = self.tok.eos_token_id
                if getattr(safe_gen, "eos_token_id", None) is None:
                    safe_gen.eos_token_id = self.tok.eos_token_id
                self.hf.generation_config = safe_gen
            except Exception as e:
                print(f"[asr] WARN: could not set safe GenerationConfig: {e}")

            # language tokens
            self.lang_to_id = {}
            for code in (
                "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha he hi hr ht hu hy id "
                "is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so "
                "sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue lg bas basaa"
            ).split():
                tid = self.tok.convert_tokens_to_ids(f"<|{code}|>")
                if isinstance(tid, int) and tid >= 0:
                    self.lang_to_id[code] = tid

            self.backend = "hf"
            print(f"[asr] using HF Whisper at: {use_hf}")
            return

        if use_ct2 is not None and CT2WhisperModel is not None:
            self.ct2 = CT2WhisperModel(str(use_ct2), device="cuda", compute_type="float16")
            self.backend = "ct2"
            print(f"[asr] using CT2 Whisper at: {use_ct2}")
            return

        raise RuntimeError("No Whisper backend available (HF bundle not found).")

    @staticmethod
    def _pcm16_to_float(wav16k: bytes) -> np.ndarray:
        return np.frombuffer(wav16k, dtype=np.int16).astype(np.float32) / 32768.0

    def _detect_lang_hf(self, feats):
        if not getattr(self, "lang_to_id", None):
            return "unk", 0.0
        sot = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        nots = self.tok.convert_tokens_to_ids("<|notimestamps|>")
        dec = torch.tensor([[sot, nots]], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out = self.hf(input_features=feats, decoder_input_ids=dec)
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        code, tid = max(self.lang_to_id.items(), key=lambda kv: float(probs[kv[1]].item()))
        return code, float(probs[tid].item())

    def _forced_ids(self, lang_code: str | None):
        try:
            ids, pos = [], 0
            for tok in [
                "<|startoftranscript|>",
                (
                    f"<|{lang_code}|>"
                    if (lang_code and self.tok.convert_tokens_to_ids(f"<|{lang_code}|>") != self.tok.unk_token_id)
                    else None
                ),
                "<|transcribe|>",
                "<|notimestamps|>",
            ]:
                if tok is None:
                    continue
                tid = self.tok.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append((pos, tid))
                    pos += 1
            return ids or None
        except Exception:
            return None

    def transcribe(self, wav16k: bytes):
        pcm = self._pcm16_to_float(wav16k)
        feats = (
            self.feat(audio=pcm, sampling_rate=16000, return_tensors="pt")
            .input_features.to(self.device)
            .to(self.hf.dtype)
        )
        code, conf = self._detect_lang_hf(feats)
        forced = self._forced_ids(code if code != "unk" else None)
        try:
            self.hf.generation_config.forced_decoder_ids = forced
        except Exception:
            self.hf.generation_config.forced_decoder_ids = None

        max_target = int(getattr(self.hf.config, "max_target_positions", 448) or 448)
        forced_len = len(forced) if forced else 2
        safe_new = max(1, min(400, max_target - forced_len - 1))

        with torch.inference_mode():
            out = self.hf.generate(
                feats,
                do_sample=False,
                num_beams=1,
                max_new_tokens=safe_new,
                pad_token_id=self.tok.eos_token_id,
                eos_token_id=self.tok.eos_token_id,
            )

        text = self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        code = "lg" if (code or "unk").lower() in BASAA_ALIASES else (code or "unk").lower()
        return text, code, conf
