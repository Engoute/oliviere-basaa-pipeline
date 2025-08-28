# FILE: app/vision_llava.py
from __future__ import annotations
import io, json, time
from typing import List, Optional
from PIL import Image
import torch

from transformers import (
    AutoProcessor, AutoTokenizer,
    LlavaNextVideoForConditionalGeneration,
)

STRICT_FRENCH_INSTRUCTION = (
    "Tu es un assistant de vision. Réponds UNIQUEMENT en français, "
    "en 1 à 2 phrases courtes, sans métadonnées ni balises. "
    "Décris uniquement ce qui est VISIBLE. "
    "N'invente pas de détails. Si tu n'es pas sûr, réponds: «Je ne suis pas sûr.»"
)

def _maybe_load_chat_template(tok: AutoTokenizer, root: str):
    if getattr(tok, "chat_template", None):
        return
    try:
        import os, json
        p = os.path.join(root, "chat_template.json")
        if os.path.isfile(p):
            data = json.load(open(p, "r", encoding="utf-8"))
            tmpl = data.get("chat_template") or data.get("template") or (data if isinstance(data, str) else None)
            if isinstance(tmpl, dict):
                tmpl = tmpl.get("template")
            if isinstance(tmpl, str) and tmpl.strip():
                tok.chat_template = tmpl
    except Exception:
        pass

class LLaVAVideo:
    def __init__(self, local_dir: str, device: Optional[str] = None):
        self.local_dir = local_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True

        self.processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True, local_files_only=True)
        self.tokenizer = getattr(self.processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        _maybe_load_chat_template(self.tokenizer, local_dir)

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True,
            torch_dtype=self.dtype, device_map="auto"
        ).eval()

    def _build_prompt(self, question_fr: str) -> str:
        conv = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": f"{STRICT_FRENCH_INSTRUCTION}\n\nQuestion: {question_fr}"},
            ],
        }]
        apply = getattr(self.tokenizer, "apply_chat_template", None) or getattr(self.processor, "apply_chat_template", None)
        if apply:
            try:
                return apply(conv, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return apply(conv, add_generation_prompt=True)
        return f"<s>[INST] {STRICT_FRENCH_INSTRUCTION}\n\nQuestion: {question_fr} [/INST]"

    @torch.no_grad()
    def describe_frames(self, frames: List[Image.Image], question_fr: str, max_new_tokens: int = 96) -> str:
        """
        frames: list of PIL RGB frames (keep it short, e.g., 6–12)
        returns: French plain text
        """
        prompt = self._build_prompt(question_fr)
        # Prefer treating frames as a single video; processors sometimes accept either shape
        try:
            inputs = self.processor(text=prompt, videos=frames, return_tensors="pt")
        except Exception:
            inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")

        pix = inputs.get("pixel_values_videos")
        if isinstance(pix, list):
            pix = torch.stack(pix, dim=0)
        input_ids = inputs.get("input_ids") or self.tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.model.device)
        pix = pix.to(self.model.device, dtype=self.model.dtype)

        out = self.model.generate(
            input_ids=input_ids,
            pixel_values_videos=pix,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # ← no sampling to curb hallucinations
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        for m in ("Assistant:", "assistant:", "ASSISTANT:"):
            if m in text:
                text = text.split(m, 1)[-1].strip()
        return text

    @torch.no_grad()
    def describe_image(self, img: Image.Image, question_fr: str) -> str:
        # treat image as a short “video” (repeat few frames)
        frames = [img.convert("RGB")] * 6
        return self.describe_frames(frames, question_fr)
