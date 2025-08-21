# FILE: app/utils_audio.py
import io, struct
import numpy as np
from scipy.signal import resample_poly

def pcm16le_bytes_to_float32_mono(b: bytes) -> np.ndarray:
    return (np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)

def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    from math import gcd
    g = gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(x.astype(np.float32, copy=False), up, down).astype(np.float32)

def wav_bytes_from_float32(x: np.ndarray, sr: int) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype("<i2")

    buf = io.BytesIO()
    # RIFF header
    data_bytes = x16.tobytes()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data_bytes)))
    buf.write(b"WAVE")

    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))

    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data_bytes)))
    buf.write(data_bytes)
    return buf.getvalue()
