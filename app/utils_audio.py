# FILE: app/utils_audio.py
import numpy as np

def pcm16le_bytes_to_float32_mono(b: bytes) -> np.ndarray:
    # input: mono PCM16LE
    return (np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)
