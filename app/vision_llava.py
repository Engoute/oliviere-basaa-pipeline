# FILE: app/vision_llava.py
from __future__ import annotations
import io, json, time, os
from typing import List, Optional
from PIL import Image
import torch

from transformers import (
    AutoProcessor, AutoTokenizer,
    LlavaNextVideoForConditionalGeneration,
)

STRICT_FRENCH_INSTRUCTION = (
    "Tu es un assistant de vision. Réponds en français en 1 à 2 phrases courtes, "
    "sans métadonnées ni balises. Décris uniquement ce qui est VISIBLE, de façon précise. "
    "N'invente pas de détails. Si l'image est entièrement noire/blanche, très floue "
    "ou ne montre rien de discernable, tu peux répondre : «Je ne suis pas sûr.»"
)

SOFT_FRENCH_INSTRUCTION = (
    "Décris brièvement et précisément ce que tu vois sur l'image. "
    "Réponds en français en 1 à 2 phrases naturelles."
)

def _maybe_load_chat_template(tok: AutoTokenizer, root: str):
    if getattr(tok, "chat_template", None):
        return
    try:
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

    def _build_prompt(self, question_fr: str, strict: bool = True) -> str:
        instr = STRICT_FRENCH_INSTRUCTION if strict else SOFT_FRENCH_INSTRUCTION
        conv = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": f"{instr}\n\nQuestion: {question_fr}"},
            ],
        }]
        apply = getattr(self.tokenizer, "apply_chat_template", None) or getattr(self.processor, "apply_chat_template", None)
        if apply:
            try:
                return apply(conv, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return apply(conv, add_generation_prompt=True)
        return f"<s>[INST] {instr}\n\nQuestion: {question_fr} [/INST]"

    def _prep_inputs(self, frames: List[Image.Image], prompt: str):
        # Try “videos=frames”; some processors also accept videos=[frames]
        try:
            inputs = self.processor(text=prompt, videos=frames, return_tensors="pt")
        except Exception:
            inputs = self.processor(text=prompt, videos=[frames], return_tensors="pt")

        # Which pixel key did we get?
        pixel_key = None
        if "pixel_values_videos" in inputs:
            pixel_key = "pixel_values_videos"
            pix = inputs["pixel_values_videos"]
        elif "pixel_values" in inputs:
            pixel_key = "pixel_values"
            pix = inputs["pixel_values"]
        else:
            raise RuntimeError("Processor did not return pixel values")

        if isinstance(pix, list):
            pix = torch.stack(pix, dim=0)

        input_ids = inputs.get("input_ids", None)
        if (input_ids is None) or (hasattr(input_ids, "nelement") and input_ids.nelement() == 0):
            toks = self.tokenizer(prompt, return_tensors="pt")
            input_ids = toks["input_ids"]

        input_ids = input_ids.to(self.model.device)
        pix = pix.to(self.model.device, dtype=self.model.dtype)
        return input_ids, pix, pixel_key

    @torch.no_grad()
    def _generate(self, input_ids, pix, pixel_key: str, max_new_tokens: int, sample: bool):
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # attach pixels under the exact key name the model expects
        gen_kwargs[pixel_key] = pix

        if sample:
            gen_kwargs.update(dict(do_sample=True, temperature=0.2, top_p=0.9))
        else:
            gen_kwargs.update(dict(do_sample=False, temperature=0.0, top_p=1.0))

        out = self.model.generate(**gen_kwargs)
        return out

    def _post(self, out, prompt: str) -> str:
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        for m in ("Assistant:", "assistant:", "ASSISTANT:"):
            if m in text:
                text = text.split(m, 1)[-1].strip()
        return text

    @torch.no_grad()
    def describe_frames(self, frames: List[Image.Image], question_fr: str, max_new_tokens: int = 96) -> str:
        # Pass 1: strict, deterministic
        prompt = self._build_prompt(question_fr, strict=True)
        input_ids, pix, pixel_key = self._prep_inputs(frames, prompt)
        out = self._generate(input_ids, pix, pixel_key, max_new_tokens=max_new_tokens, sample=False)
        text = self._post(out, prompt)

        low = text.lower()
        if "je ne suis pas sûr" in low or "je ne suis pas sur" in low:
            # Pass 2: softer, allow tiny sampling to escape the safety basin
            prompt2 = self._build_prompt(question_fr, strict=False)
            input_ids2, pix2, pixel_key2 = self._prep_inputs(frames, prompt2)
            out2 = self._generate(input_ids2, pix2, pixel_key2, max_new_tokens=max_new_tokens, sample=True)
            text2 = self._post(out2, prompt2)
            if text2.strip():
                text = text2
        return text

    @torch.no_grad()
    def describe_image(self, img: Image.Image, question_fr: str) -> str:
        frames = [img.convert("RGB")] * 6
        return self.describe_frames(frames, question_fr)
