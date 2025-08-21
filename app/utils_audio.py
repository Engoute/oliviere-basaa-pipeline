# FILE: app/utils_audio.py
import io, struct
import numpy as np

def pcm16le_bytes_to_float32_mono(b: bytes) -> np.ndarray:
    return (np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0)

def _resample_torchaudio(x: np.ndarray, sr_from: int, sr_to: int):
    try:
        import torch, torchaudio
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        t = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        y = torchaudio.functional.resample(t, sr_from, sr_to)
        return y.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        return None

def _resample_scipy(x: np.ndarray, sr_from: int, sr_to: int):
    try:
        from math import gcd
        from scipy.signal import resample_poly  # optional
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        g = gcd(sr_from, sr_to)
        up, down = sr_to // g, sr_from // g
        return resample_poly(x, up, down).astype(np.float32)
    except Exception:
        return None

def _resample_numpy_interp(x: np.ndarray, sr_from: int, sr_to: int):
    # Fallback: linear interpolation in NumPy
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * float(sr_to) / float(sr_from)))
    if n_out <= 1 or len(x) <= 1:
        return x
    t_src = np.linspace(0.0, 1.0, num=len(x), dtype=np.float64)
    t_dst = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
    y = np.interp(t_dst, t_src, x).astype(np.float32)
    return y

def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    # Try torchaudio (fast, already in requirements)
    y = _resample_torchaudio(x, sr_from, sr_to)
    if y is not None:
        return y
    # Try SciPy if present
    y = _resample_scipy(x, sr_from, sr_to)
    if y is not None:
        return y
    # Fallback: NumPy interpolation
    return _resample_numpy_interp(x, sr_from, sr_to)

def wav_bytes_from_float32(x: np.ndarray, sr: int) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype("<i2")

    buf = io.BytesIO()
    data_bytes = x16.tobytes()

    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data_bytes)))
    buf.write(b"WAVE")

    # fmt chunk (PCM, mono, 16-bit)
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))

    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data_bytes)))
    buf.write(data_bytes)

    return buf.getvalue()
