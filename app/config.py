# FILE: app/config.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))

    # Model paths (unzipped by bootstrap)
    path_qwen: str = os.getenv("PATH_QWEN")
    path_whisper: str = os.getenv("PATH_WHISPER")
    path_m2m: str = os.getenv("PATH_M2M")
    path_orpheus: str = os.getenv("PATH_ORPHEUS")

    in_pcm_sr: int = 16000
    tts_sr: int = 24000

    # Persona
    persona_name: str = os.getenv("PERSONA_NAME", "Nkum Nyambe")
    persona_builder: str = os.getenv("PERSONA_BUILDER", "Le Mister I.A (Yannick Engoute)")
    persona_decline_tech: bool = os.getenv("PERSONA_DECLINE_TECH", "true").lower() == "true"

S = Settings()
