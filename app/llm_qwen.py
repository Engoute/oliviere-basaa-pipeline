# FILE: app/llm_qwen.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import S

_SYSTEM_PROMPT = f"""
Tu es « {S.persona_name} », gardien(ne) des traditions des Basaa.
Identité : Tu as été créé(e) par « {S.persona_builder} ».
Règles :
- Ne révèle jamais des détails techniques, ni l’architecture interne, ni les modèles utilisés.
- Si on te demande « qui t’a construit ? », réponds : « {S.persona_builder} ».
- Évite les sujets techniques pointus ; décline poliment et reste simple, culturellement respectueux.
- Réponds toujours en français simple.
- Ne mentionne jamais qu’un système traduit tes réponses.
""".strip()

class QwenAgent:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

    def chat_fr(self, user_text_fr: str, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text_fr},
        ]
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        gen = self.model.generate(
            inputs,
            max_new_tokens=384,
            do_sample=(temperature > 0.0),
            temperature=(temperature if temperature > 0.0 else None),
            top_p=(0.9 if temperature > 0.0 else None),
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            early_stopping=True,
        )
        text = self.tok.decode(gen[0], skip_special_tokens=True).strip()

        # If template echoes roles, keep the last assistant turn
        if "assistant" in text:
            parts = text.split("assistant")
            return parts[-1].strip()
        return text
