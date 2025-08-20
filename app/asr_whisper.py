# /workspace/app/asr_whisper.py
from pathlib import Path
import numpy as np, torch

try:
    from faster_whisper import WhisperModel as CT2WhisperModel
except Exception:
    CT2WhisperModel = None

from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASAA_ALIASES = {"lg","bas","basaa"}  # normalize any of these → "lg" internally

class ASR:
    def __init__(self, base_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        p = Path(base_path) if base_path else None

        # Prefer CT2 if present
        if p and (p/"model.bin").exists() and CT2WhisperModel is not None:
            self.ct2 = CT2WhisperModel(str(p), device="cuda", compute_type="float16")
            self.backend = "ct2"
            return

        # HF merged checkpoint (your bundle)
        if p and ((p/"model.safetensors.index.json").exists() or any(p.glob("model-*.safetensors"))):
            self.proc  = WhisperProcessor.from_pretrained(str(p), subfolder="processor", local_files_only=True)
            self.hf    = WhisperForConditionalGeneration.from_pretrained(
                str(p), local_files_only=True, torch_dtype=self.dtype, low_cpu_mem_usage=True
            ).to(self.device).eval()
            self.tok   = self.proc.tokenizer
            self.lang_to_id = {}
            for code in (
                "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha he hi hr ht hu hy id "
                "is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so "
                "sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue lg bas basaa"
            ).split():
                tid = self.tok.convert_tokens_to_ids(f"<|{code}|>")
                if isinstance(tid,int) and tid>=0: self.lang_to_id[code]=tid
            self.backend = "hf"
            return

        # Last resort: stock CT2 (keeps API alive)
        if CT2WhisperModel is not None:
            self.ct2 = CT2WhisperModel("large-v3", device="cuda", compute_type="float16")
            self.backend = "ct2"
        else:
            raise RuntimeError("No Whisper backend available.")

    @staticmethod
    def _pcm16_to_float(wav16k: bytes) -> np.ndarray:
        return (np.frombuffer(wav16k, dtype=np.int16).astype(np.float32) / 32768.0)

    def _detect_lang_hf(self, feats):
        if not self.lang_to_id: return "unk", 0.0
        sot  = self.tok.convert_tokens_to_ids("<|startoftranscript|>")
        nots = self.tok.convert_tokens_to_ids("<|notimestamps|>")
        dec  = torch.tensor([[sot, nots]], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            out   = self.hf(input_features=feats, decoder_input_ids=dec)
            probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        code, tid = max(self.lang_to_id.items(), key=lambda kv: float(probs[kv[1]].item()))
        return code, float(probs[tid].item())

    def transcribe(self, wav16k: bytes):
        if getattr(self, "backend", None) == "ct2":
            pcm = self._pcm16_to_float(wav16k)
            segs, info = self.ct2.transcribe(pcm, language=None, beam_size=1, vad_filter=True, temperature=0.0)
            text = " ".join(s.text.strip() for s in segs).strip()
            code = (info.language or "unk").lower()
            code = "lg" if code in BASAA_ALIASES else code
            conf = float(getattr(info,"language_probability",0.0) or 0.0)
            return text, code, conf

        # HF merged path
        pcm   = self._pcm16_to_float(wav16k)
        feats = self.proc(audio=pcm, sampling_rate=16000, return_tensors="pt").input_features.to(self.device).to(self.hf.dtype)

        code, conf = self._detect_lang_hf(feats)
        # Don't force a language unless the token exists
        try:
            forced = self.proc.get_decoder_prompt_ids(language=code if code!="unk" else None, task="transcribe")
            self.hf.generation_config.forced_decoder_ids = forced
        except Exception:
            self.hf.generation_config.forced_decoder_ids = self.proc.get_decoder_prompt_ids(language=None, task="transcribe")

        with torch.inference_mode():
            out = self.hf.generate(
                feats, do_sample=False, num_beams=1,
                max_new_tokens=min(400, getattr(self.hf.config, "max_target_positions", 448)-3),
                pad_token_id=self.tok.eos_token_id, eos_token_id=self.tok.eos_token_id,
            )
        text = self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        code = "lg" if (code or "unk").lower() in BASAA_ALIASES else (code or "unk").lower()
        return text, code, conf
