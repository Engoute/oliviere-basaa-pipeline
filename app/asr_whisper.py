# /workspace/app/asr_whisper.py
from pathlib import Path
import numpy as np
import torch

# ---- Optional CT2 backend (only used if a CT2 bundle is present) ----
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


class ASR:
    def __init__(self, base_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        p = Path(base_path) if base_path else None

        # ---- Prefer CT2 if present (model.bin) and CT2 is installed ----
        if p and (p / "model.bin").exists() and CT2WhisperModel is not None:
            self.ct2 = CT2WhisperModel(str(p), device="cuda", compute_type="float16")
            self.backend = "ct2"
            return

        # ---- HF merged checkpoint (your bundle) ----
        if p and ((p / "model.safetensors.index.json").exists() or any(p.glob("model-*.safetensors"))):
            if _WHISPER_NATIVE and WhisperProcessor is not None:
                proc = WhisperProcessor.from_pretrained(
                    str(p), subfolder="processor", local_files_only=True
                )
                self.tok = proc.tokenizer
                self.feat = proc.feature_extractor
            else:
                # Manual processor: tokenizer + feature_extractor
                self.tok = WhisperTokenizer.from_pretrained(
                    str(p), local_files_only=True
                )
                # try subfolder="processor" first (your bundle layout), then root
                try:
                    self.feat = WhisperFeatureExtractor.from_pretrained(
                        str(p), subfolder="processor", local_files_only=True
                    )
                except Exception:
                    self.feat = WhisperFeatureExtractor.from_pretrained(
                        str(p), local_files_only=True
                    )

            self.hf = WhisperForConditionalGeneration.from_pretrained(
                str(p), local_files_only=True, torch_dtype=self.dtype, low_cpu_mem_usage=True
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
            return

        # ---- Last resort: stock CT2 (keeps API alive) ----
        if CT2WhisperModel is not None:
            self.ct2 = CT2WhisperModel("large-v3", device="cuda", compute_type="float16")
            self.backend = "ct2"
        else:
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
        Returns a list of (position, token_id) pairs or None.
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

        # Set forced decoder ids (either via real processor or manual)
        forced = None
        # If native processor exists, try using its method through a tiny shim
        if _WHISPER_NATIVE and hasattr(self, "proc"):  # not used currently, kept for clarity
            try:
                forced = self.proc.get_decoder_prompt_ids(
                    language=code if code != "unk" else None, task="transcribe"
                )
            except Exception:
                forced = None
        if forced is None:
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
