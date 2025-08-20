import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .config import S

# We stick to HF Transformers here (H200 is plenty fast)
class M2M:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        # Basaa tag
        self.basaa_code = "lg"  # as you specified

    def _generate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        tok = self.tok
        inputs = tok(text, return_tensors="pt").to(self.model.device)
        # set language tokens as M2M requires
        forced_bos_token_id = tok.get_lang_id(tgt_lang)
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
        )
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    def to_fr(self, text: str, src_lang: str) -> str:
        # src could be lg or en/fr/other; if already fr, just return
        if src_lang == "fr": return text
        if src_lang == self.basaa_code:
            return self._generate(text, src_lang, "fr")
        elif src_lang == "en":
            return self._generate(text, "en", "fr")
        else:
            # unknown → try direct to fr
            return self._generate(text, src_lang, "fr")

    def to_lg(self, text: str, src_lang: str) -> str:
        if src_lang == self.basaa_code: return text
        # Prefer FR→LG; English first passes through FR if needed
        if src_lang == "fr":
            return self._generate(text, "fr", self.basaa_code)
        elif src_lang == "en":
            fr = self._generate(text, "en", "fr")
            return self._generate(fr, "fr", self.basaa_code)
        else:
            # Fallback: try target lg directly
            return self._generate(text, src_lang, self.basaa_code)
