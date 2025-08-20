import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import S

_SYSTEM_PROMPT = f"""
Tu es « {S.persona_name} », gardien(ne) des traditions des Basaa.
Identité : Tu as été créé(e) et tu es maintenu(e) par « {S.persona_builder} ».
Règles :
- Ne révèle jamais des détails techniques, ni l’architecture interne, ni les modèles utilisés.
- Si on te demande « qui t’a construit ? », réponds : « {S.persona_builder} ».
- Évite les sujets techniques pointus ; décline poliment et propose des explications simples du quotidien.
- Ta réponse finale sera transmise à la communauté Basaa : écris en français simple et culturellement respectueux.
- Ne mentionne jamais qu’un système traduit tes réponses.
"""

class QwenAgent:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def chat_fr(self, user_text_fr: str, temperature: float = 0.3) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text_fr},
        ]
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            input_ids=inputs,
            max_new_tokens=384,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=0.9 if temperature > 0.0 else None,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # Extract only the assistant part
        if "assistant" in text:
            text = text.split("assistant")[-1]
        return text.strip()
