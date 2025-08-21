# /workspace/app/config.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))

    # bundle paths (already unzipped by bootstrap)
    path_qwen: str = os.getenv("PATH_QWEN")
    # HF Whisper (NOT CT2). Bootstrap also creates /data/models/whisper_hf_resolved.
    path_whisper: str = os.getenv("PATH_WHISPER", "/data/models/whisper_hf")
    path_m2m: str = os.getenv("PATH_M2M")
    path_orpheus: str = os.getenv("PATH_ORPHEUS")

    # audio
    in_pcm_sr: int = 16000   # your C# client: 100ms @ 16kHz -> 3200 bytes
    tts_sr: int = 24000

    # persona
    persona_name: str = os.getenv("PERSONA_NAME", "Nkum Nyambe")
    persona_builder: str = os.getenv("PERSONA_BUILDER", "Le Mister I.A (Yannick Engoute)")
    persona_decline_tech: bool = os.getenv("PERSONA_DECLINE_TECH", "true").lower() == "true"

S = Settings()
