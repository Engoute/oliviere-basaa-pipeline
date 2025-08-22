# FILE: app/config.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))

    path_whisper: str = os.getenv("PATH_WHISPER")
    path_m2m: str = os.getenv("PATH_M2M")
    path_orpheus: str = os.getenv("PATH_ORPHEUS")  # NEW

    in_pcm_sr: int = 16000
    tts_sr: int = int(os.getenv("TTS_SR", "24000"))

S = Settings()
