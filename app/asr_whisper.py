# /workspace/app/asr_whisper.py
from pathlib import Path
import numpy as np
import torch

# ---- Optional CT2 backend (used only if a CT2 bundle with model.bin is present) ----
try:
    from faster_whisper import WhisperModel as CT2WhisperModel  # type: ignore
except Exception:
    CT2WhisperModel = None

# ---- Try native Whisper classes; fall back to auto-classes + manual glue ----
try:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
        WhisperFeatureExtractor,
    )  # type: ignore
    _WHISPER_NATIVE = True
except Exception:
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoTokenizer,
        AutoFeatureExtractor,
    )  # type: ignore
    WhisperForConditionalGeneration = AutoModelForSpeechSeq2Seq  # type: ignore
    WhisperTokenizer = AutoTokenizer  # type: ignore
    WhisperFeatureExtractor = AutoFeatureExtractor  # type: ignore
    WhisperProcessor = None  # type: ignore
    _WHISPER_NATIVE = False


BASAA_ALIASES = {"lg", "bas", "basaa"}  # normalize any of these → "lg"


def _resolve_whisper_dir(base: Path):
    """
    Return (use_path, kind) where kind ∈ {"hf","ct2"} by scanning nested folders.
    Priority: CT2 model.bin if present, otherwise HF (config.json + tokenizer/processor).
    """
    if not base or not base.exists():
        return None, None

    # 1) CT2 at root?
    if (base / "model.bin").exists():
        return base, "ct2"

    # 2) HF at root?
    if (base / "config.json").exists():
        # consider it HF if tokenizer or processor files are around
        if (base / "tokenizer.json").exists() or (base / "tokenizer_config.json").exists() or (base / "processor").exists():
            return base, "hf"

    # 3) Scan nested: prefer CT2 model.bin
    ct2_bins = list(base.rglob("model.bin"))
    if ct2_bins:
        return ct2_bins[0].parent, "ct2"

    # 4) Scan nested HF: config.json with tokenizer/processor nearby
    cand = None
    best_score = -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        has_tok = (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()
        has_proc = (d / "processor").exists()
        # more evidence -> higher score
        score = (2 if has_tok else 0) + (1 if has_proc else 0)
        if score > best_score:
            cand, best_score = d, score
    if cand is not None:
        return cand, "hf"

    return None, None


class ASR:
    def __init__(self, base_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        p = Path(base_path) if base_path else None

        use, kind = _resolve_whisper_dir(p) if p else (None, None)

        # ---- Prefer CT2 if detected and CT2 is installed ----
        if kind == "ct2":
            if CT2WhisperModel is None:
                raise RuntimeError("CT2 bundle found but faster-whisper is not installed.")
            self.ct2 = CT2WhisperModel(str(use), device="cuda", compute_type="float16")
            self.backend = "ct2"
            print(f"[asr] using CT2 model at: {use}")
            return

        # ---- HF merged checkpoint (your bundle) ----
        if kind == "hf":
            # Processor (native or manual)
            if _WHISPER_NATIVE and WhisperProcessor is not None:
                try:
                    self.proc = WhisperProcessor.from_pretrained(
                        str(use), subfolder="processor", local_files_only=True
                    )
                    self.tok = self.proc.tokenizer
                    self.feat = self.proc.feature_extractor
                except Exception:
                    # fallback: try root if no /processor
                    self.proc = WhisperProcessor.from_pretrained(
                        str(use), local_files_only=True
                    )
                    self.tok = self.proc.tokenizer
                    self.feat = self.proc.feature_extractor
            else:
                self.tok = WhisperTokenizer.from_pretrained(str(use), local_files_only=True)
                # prefer subfolder=processor; fallback root
                try:
                    self.feat = WhisperFeatureExtractor.from_pretrained(
                        str(use), subfolder="processor", local_files_only=True
                    )
                except Exception:
                    self.feat = WhisperFeatureExtractor.from_pretrained(
                        str(use), local_files_only=True
                    )

            self.hf = WhisperForConditionalGeneration.from_pretrained(
                str(use),
                local_files_only=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()

            # Build language-token map once
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
            print(f"[asr] using HF model at: {use}")
            return

        # ---- Nothing found ----
        raise RuntimeError("No Whisper backend available.")

    @staticmethod
    def _pcm16_to_float(wav16k: bytes) -> np.ndarray:
        return (np.frombuffer(wav16k, dtype=np.int16).astype(np.float32) / 32768.0)

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
        """
        Build forced decoder ids similar to WhisperProcessor.get_decoder_prompt_ids().
        Layout: <|startoftranscript|> [<|xx|>] <|transcribe|> <|notimestamps|>
        """
        try:
            ids = []
            pos = 0
            for tok in [
                "<|startoftranscript|>",
                (f"<|{lang_code}|>" if (lang_code and self.tok.convert_tokens_to_ids(f"<|{lang_code}|>") != self.tok.unk_token_id) else None),
                "<|transcribe|>",
                "<|notimestamps|>",
            ]:
                if tok is None:
                    continue
                tid = self.tok.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    ids.append((pos, tid))
                    pos += 1
            return ids if ids else None
        except Exception:
            return None

    def transcribe(self, wav16k: bytes):
        # ---- CT2 path ----
        if getattr(self, "backend", None) == "ct2":
            pcm = self._pcm16_to_float(wav16k)
            segs, info = self.ct2.transcribe(
                pcm, language=None, beam_size=1, vad_filter=True, temperature=0.0
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            code = (info.language or "unk").lower()
            code = "lg" if code in BASAA_ALIASES else code
            conf = float(getattr(info, "language_probability", 0.0) or 0.0)
            return text, code, conf

        # ---- HF merged path ----
        pcm = self._pcm16_to_float(wav16k)
        feats = (
            self.feat(audio=pcm, sampling_rate=16000, return_tensors="pt")
            .input_features.to(self.device)
            .to(self.hf.dtype)  # match model dtype (fp16 on GPU)
        )

        code, conf = self._detect_lang_hf(feats)

        # Set forced decoder ids (manual)
        forced = self._forced_ids(code if code != "unk" else None)
        try:
            self.hf.generation_config.forced_decoder_ids = forced
        except Exception:
            self.hf.generation_config.forced_decoder_ids = None

        # Safe token budget: respect max_target_positions
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

        # Decode
        text = self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        code = "lg" if (code or "unk").lower() in BASAA_ALIASES else (code or "unk").lower()
        return text, code, conf
