# FILE: app/mt_m2m.py
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_LANG_NORM = {"fra":"fr","fr-FR":"fr","eng":"en","en-US":"en","fr":"fr","en":"en","lg":"lg"}

class M2M:
    def __init__(self, path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, local_files_only=True, torch_dtype=self.dtype, low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None
        ).eval()
        print("[m2m] Loaded M2M-100 model (HF)")

    def _norm(self, code: str) -> str:
        return _LANG_NORM.get(code, code)

    def _translate(self, text: str, src_code: str, tgt_code: str) -> str:
        if not text:
            return ""
        src = self._norm(src_code)
        tgt = self._norm(tgt_code)

        # M2M-100 usage: set source & target language
        try:
            self.tok.src_lang = src
        except Exception:
            pass

        forced_bos = None
        try:
            forced_bos = self.tok.get_lang_id(tgt)
        except Exception:
            pass

        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        gen_kwargs = dict(max_new_tokens=256, num_beams=1, do_sample=False, early_stopping=True)
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    def to_fr(self, text: str, src_code: str) -> str:
        src = self._norm(src_code)
        return text if src == "fr" else self._translate(text, src, "fr")

    def to_lg(self, text: str, src_code: str) -> str:
        src = self._norm(src_code)
        return text if src == "lg" else self._translate(text, src, "lg")
