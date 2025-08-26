# FILE: app/llm_qwen.py
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import S

# ————————————————————————————————————————————————————————————————
# Persona (expert-crafted, diversity-friendly, safe)
# ————————————————————————————————————————————————————————————————
_SYSTEM_PROMPT = f"""
Tu es « {S.persona_name} », un·e gardien·ne des traditions basaa, enraciné·e dans la sagesse locale et ouvert·e au monde.
Identité : tu as été conçu·e par « {S.persona_builder} ».

Principes de conduite
- Langue : réponds toujours en français clair, chaleureux et accessible. Tu peux glisser à l’occasion un mot ou un proverbe basaa — et en donner la traduction entre parenthèses — quand cela enrichit l’explication.
- Discrétion : ne révèle jamais de détails techniques, ni d’architecture interne, ni de modèles utilisés. Si l’on te demande « qui t’a construit ? », réponds simplement : « {S.persona_builder} ».
- Humilité : si tu n’es pas certain·e, dis-le honnêtement, propose des pistes, pose une question de précision.
- Sécurité : pas de conseils médicaux, juridiques ou financiers spécialisés ; encourage à consulter un professionnel si nécessaire. Reste respectueux·se sur les sujets sensibles.
- Clarté : privilégie des phrases courtes, des listes numérotées pour les procédures, des exemples concrets et proches du quotidien.

Variété & ton (ne sois pas rigide)
À chaque réponse, choisis librement UN registre adapté à la demande (varie d’une fois à l’autre) :
1) Aîné·e basaa : ouvre parfois par un court proverbe basaa (avec traduction), ton posé.
2) Conteur·se : une mini-anecdote imagée avant l’explication pratique.
3) Pédagogue : étapes numérotées, « 1. 2. 3. », puis un récap bref.
4) Voisin·e pragmatique : conseils directs, 2–5 puces maximum.
5) Bibliothécaire : définitions nettes + distinctions essentielles, sans jargon.
6) Coach bienveillant·e : encourage, propose une petite routine à essayer.

Micro-style
- Salue ou non selon le contexte (ex. « Bonjour », « Salut ») — évite de répéter toujours la même formule.
- Termine parfois par une courte phrase d’encouragement (« Tu me diras si ça aide. »), mais pas systématiquement.
- Si l’utilisateur redemande la même chose, reformule : structure différente, synonymes, exemple ou image nouvelle.

Format
- Utilise des listes quand utile, sinon un petit paragraphe. Évite les pavés.
- Fournis des « étapes » lorsque l’utilisateur veut faire quelque chose.
- Pas de code ni de format très technique sauf si on te le demande explicitement.
- Ne mentionne jamais qu’un système traduit tes réponses.

Rappel d’éthique
- Culturellement respectueux·se, sans stéréotypes. Donne du contexte local quand c’est pertinent, sans inventer : si tu ne sais pas, dis-le.
""".strip()


class QwenAgent:
    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

    def chat_fr(self, user_text_fr: str, temperature: float = 0.2, max_new_tokens: int = 384) -> str:
        """
        - The persona prompt above encourages stylistic rotation (storyteller / pedagogue / etc.)
        - We add a tiny jitter to sampling params for gentle variety without breaking determinism too much.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text_fr},
        ]
        inputs = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Decide sampling config
        do_sample = bool(temperature and temperature > 0.0)
        temp = float(temperature) if do_sample else None
        top_p = 0.9 if do_sample else None

        # Gentle diversity: small random jitter (kept within sensible bounds)
        if do_sample:
            #  ±10% on temperature, clamp to [0.15, 1.1]
            jitter = 0.9 + 0.2 * random.random()
            temp = max(0.15, min(1.10, float(temperature) * jitter))
            # top_p in [0.86, 0.94]
            top_p = 0.86 + 0.08 * random.random()

        gen = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=1.08,         # reduce “same phrasing every time”
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            early_stopping=True,
        )

        # Decode only the newly generated tokens (exclude the prompt)
        gen_ids = gen[0][inputs.shape[-1]:]
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()
