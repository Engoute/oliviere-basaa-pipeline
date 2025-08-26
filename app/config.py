# FILE: app/config.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))

    # Back-compat (Basaa finetune)
    path_whisper: str = os.getenv("PATH_WHISPER", "")

    # Explicit dual-model paths
    path_whisper_basaa: str = os.getenv("PATH_WHISPER_BASAA", os.getenv("PATH_WHISPER", ""))
    path_whisper_general: str = os.getenv("PATH_WHISPER_GENERAL", "")

    path_m2m: str = os.getenv("PATH_M2M", "")
    path_orpheus: str = os.getenv("PATH_ORPHEUS", "")
    path_qwen: str = os.getenv("PATH_QWEN", "")

    in_pcm_sr: int = 16000
    tts_sr: int = int(os.getenv("TTS_SR", "24000"))

    persona_name: str = os.getenv("PERSONA_NAME", "Nkum Nyambe")
    persona_builder: str = os.getenv("PERSONA_BUILDER", "Le Mister I.A (Yannick Engoute)")
    persona_decline_tech: bool = os.getenv("PERSONA_DECLINE_TECH", "true").lower() == "true"

S = Settings()
