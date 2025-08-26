# FILE: app/mt_m2m.py
from __future__ import annotations
from typing import Optional
import re, unicodedata
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Normalize various aliases to M2M's expected codes where possible.
# (Add more as you discover them.)
_LANG_NORM = {
    # Canonical
    "fr": "fr", "en": "en", "lg": "lg",
    # Norwegian variants
    "nn": "no", "nb": "no", "no": "no",
    # Javanese old code
    "jw": "jv",
    # Portuguese
    "pt-br": "pt", "pt_pt": "pt",
    # Chinese varieties (M2M may vary by package; keep 'zh' generic if unsure)
    "zh-cn": "zh", "zh-hans": "zh", "zh-hant": "zh", "zh-tw": "zh",
    # Common HF spellings
    "fra": "fr", "eng": "en",
}

# Very light heuristics to avoid relying on Whisper's code.
_BASAA_CHARS = set("ŋɓƁɛɔ")
_FR_DIACRITICS = set("àâäáçéèêëîïìôöòùûüúÿœæ")
_EN_STOP = {"the","and","you","your","is","are","to","of","in","for","on","with","i","we","it","that"}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def looks_like_basaa(text: str) -> bool:
    t = text.lower()
    if any(ch in t for ch in _BASAA_CHARS):
        return True
    # Light extra hints (expand as needed)
    return False

def looks_like_fr(text: str) -> bool:
    t = text.lower()
    if any(ch in t for ch in _FR_DIACRITICS):
        return True
    # French stopwords (tiny set)
    fr_stop = {"le","la","les","des","de","du","un","une","et","est","je","tu","il","elle","nous","vous","ils","elles","au","aux","dans","pour","sur"}
    words = re.findall(r"[a-zàâäáçéèêëîïìôöòùûüúÿœæ]+", t)
    return any(w in fr_stop for w in words[:20])

def looks_like_en(text: str) -> bool:
    t = text.lower()
    if not t.strip():
        return False
    # Mostly ASCII?
    ascii_ratio = sum(1 for c in t if ord(c) < 128) / max(1, len(t))
    if ascii_ratio > 0.98:
        words = re.findall(r"[a-z]+", t)
        return any(w in _EN_STOP for w in words[:20])
    return False

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

    def _norm(self, code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        c = code.lower()
        return _LANG_NORM.get(c, c)

    def resolve_safe_src(self, text: str, hint_from_client: Optional[str]) -> str:
        """
        Decide a SAFE source code for M2M (hint-first):
        1) If the client provided a hint and the tokenizer supports it -> use it
        2) Else heuristics: Basaa -> 'lg', French -> 'fr', English -> 'en'
        3) Fallback -> 'fr', then 'en', then 'lg'
        Always returns a code accepted by self.tok.get_lang_id (or a reasonable fallback).
        """
        # 1) Explicit user hint wins if supported
        h = self._norm(hint_from_client)
        if h:
            try:
                _ = self.tok.get_lang_id(h)  # throws if unsupported
                return h
            except Exception:
                pass

        # 2) Heuristics if no usable hint
        if looks_like_basaa(text): return "lg"
        if looks_like_fr(text):    return "fr"
        if looks_like_en(text):    return "en"

        # 3) Fallbacks
        for cand in ("fr", "en", "lg"):
            try:
                _ = self.tok.get_lang_id(cand)
                return cand
            except Exception:
                continue

        # Ultimate fallback (very unlikely), return English
        return "en"

    def _translate(self, text: str, src_code: str, tgt_code: str) -> str:
        if not text:
            return ""
        src = self._norm(src_code) or "fr"
        tgt = self._norm(tgt_code) or "fr"

        if src == tgt:
            return text

        # Set src language if tokenizer supports it
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
        src = self._norm(src_code) or "fr"
        return text if src == "fr" else self._translate(text, src, "fr")

    def to_lg(self, text: str, src_code: str) -> str:
        src = self._norm(src_code) or "lg"
        return text if src == "lg" else self._translate(text, src, "lg")

    def to_en(self, text: str, src_code: str) -> str:
        src = self._norm(src_code) or "en"
        return text if src == "en" else self._translate(text, src, "en")
