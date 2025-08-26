# FILE: app/utils_audio.py
import io
import struct
from typing import Tuple, Optional

import numpy as np

def _to_float32_mono(x: np.ndarray) -> np.ndarray:
    """Ensure float32 mono [-1,1]. Accepts shapes [T], [C,T], [T,1], etc."""
    x = np.asarray(x)
    if x.ndim == 1:
        mono = x.astype(np.float32, copy=False)
    else:
        # Treat first dimension as channels if ambiguous
        if x.shape[0] <= 8 and x.shape[0] < x.shape[-1]:
            # [C,T] -> mean over channels
            mono = x.mean(axis=0).astype(np.float32, copy=False)
        else:
            # [T,C] or something else -> mean over last dim
            mono = x.mean(axis=-1).astype(np.float32, copy=False)
    # Clip to safe range if it looks like PCM
    if mono.dtype != np.float32:
        mono = mono.astype(np.float32, copy=False)
    return np.clip(mono, -1.0, 1.0)

def _resample_if_needed(wav: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32, copy=False)
    try:
        # torchaudio is already in requirements; use it when available
        import torch, torchaudio
        t = torch.from_numpy(wav.astype(np.float32, copy=False)).unsqueeze(0)  # [1,T]
        r = torchaudio.functional.resample(t, sr_in, sr_out).squeeze(0).contiguous()
        return r.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        # light fallback: polyphase via numpy (ok for short clips)
        import math
        gcd = math.gcd(sr_in, sr_out)
        up, down = sr_out // gcd, sr_in // gcd
        # crude zero-order hold + decimate (keeps it dependency-free)
        idx = (np.arange(0, len(wav) * up) // up * down).astype(np.int64)
        idx = np.clip(idx, 0, len(wav) - 1)
        return wav[idx].astype(np.float32, copy=False)

def decode_audio_to_16k_float_mono(payload: bytes) -> np.ndarray:
    """
    Best-effort decoder:
      1) torchaudio.load (handles WAV/MP3/OGG/FLAC…)
      2) builtin wave module for PCM WAV
      3) raw PCM16 little-endian bytes (last resort)
    Always returns float32 mono at 16 kHz.
    """
    # 1) torchaudio
    try:
        import torchaudio
        buf = io.BytesIO(payload)
        waveform, sr = torchaudio.load(buf)   # [C,T], float32 in [-1,1] or int16 in [-32768,32767]
        wav = waveform.detach().cpu().numpy()
        wav = _to_float32_mono(wav)
        return _resample_if_needed(wav, int(sr), 16000)
    except Exception:
        pass

    # 2) pure-Python WAV (PCM) fallback
    try:
        import wave
        buf = io.BytesIO(payload)
        with wave.open(buf, "rb") as w:
            nch = w.getnchannels()
            sr  = w.getframerate()
            n   = w.getnframes()
            sampwidth = w.getsampwidth()
            raw = w.readframes(n)
        if sampwidth == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sampwidth == 1:
            # unsigned 8-bit PCM
            x = (np.frombuffer(raw, dtype="|u1").astype(np.float32) - 128.0) / 128.0
        else:
            # unsupported width -> try to interpret as 16-bit
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
        return _resample_if_needed(x, int(sr), 16000)
    except Exception:
        pass

    # 3) raw PCM16 little-endian fallback
    try:
        x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
        # Can't know original sr; assume 16k
        return x
    except Exception:
        # ultimate fallback: short silence to avoid hard crash
        return np.zeros(1600, dtype=np.float32)  # 0.1s @ 16k

def wav_bytes_from_float32(x: np.ndarray, sr: int) -> bytes:
    """Write mono PCM16 WAV bytes from float32 waveform [-1,1]."""
    x16 = np.clip(x, -1.0, 1.0)
    x16 = (x16 * 32767.0).astype("<i2")
    buf = io.BytesIO()
    def w(u32): buf.write(struct.pack("<I", u32))
    def h(u16): buf.write(struct.pack("<H", u16))
    data = x16.tobytes()
    buf.write(b"RIFF"); w(36 + len(data)); buf.write(b"WAVE")
    buf.write(b"fmt "); w(16); h(1); h(1); w(sr); w(sr*2); h(2); h(16)
    buf.write(b"data"); w(len(data)); buf.write(data)
    return buf.getvalue()
