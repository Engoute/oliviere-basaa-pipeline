# FILE: app/utils_audio.py
import io
import struct
import math
from typing import Tuple, Optional

import numpy as np


def _to_float32_mono(x: np.ndarray) -> np.ndarray:
    """Ensure float32 mono [-1,1]. Accepts shapes [T], [C,T], [T,1], etc."""
    x = np.asarray(x)
    if x.ndim == 1:
        mono = x.astype(np.float32, copy=False)
    else:
        if x.shape[0] <= 8 and x.shape[0] < x.shape[-1]:
            mono = x.mean(axis=0).astype(np.float32, copy=False)   # [C,T] -> mono
        else:
            mono = x.mean(axis=-1).astype(np.float32, copy=False)  # [T,C] -> mono
    return np.clip(mono.astype(np.float32, copy=False), -1.0, 1.0)


def _resample_if_needed(wav: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return wav.astype(np.float32, copy=False)
    try:
        import torch, torchaudio
        t = torch.from_numpy(wav.astype(np.float32, copy=False)).unsqueeze(0)  # [1,T]
        r = torchaudio.functional.resample(t, sr_in, sr_out).squeeze(0).contiguous()
        return r.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        # light polyphase-ish fallback
        g = math.gcd(sr_in, sr_out)
        up, down = sr_out // g, sr_in // g
        idx = (np.arange(0, len(wav) * up) // up * down).astype(np.int64)
        idx = np.clip(idx, 0, len(wav) - 1)
        return wav[idx].astype(np.float32, copy=False)


def decode_audio_to_16k_float_mono(payload: bytes) -> np.ndarray:
    """
    Best-effort decoder:
      1) torchaudio.load (WAV/MP3/OGG/FLAC)
      2) builtin wave for PCM WAV
      3) raw PCM16 little-endian (fallback)
    Always returns float32 mono @ 16k.
    """
    # 1) torchaudio
    try:
        import torchaudio
        buf = io.BytesIO(payload)
        waveform, sr = torchaudio.load(buf)
        wav = waveform.detach().cpu().numpy()
        wav = _to_float32_mono(wav)
        return _resample_if_needed(wav, int(sr), 16000)
    except Exception:
        pass

    # 2) pure-Python WAV
    try:
        import wave
        buf = io.BytesIO(payload)
        with wave.open(buf, "rb") as w:
            nch = w.getnchannels()
            sr = w.getframerate()
            n = w.getnframes()
            sw = w.getsampwidth()
            raw = w.readframes(n)
        if sw == 2:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sw == 1:
            x = (np.frombuffer(raw, dtype="|u1").astype(np.float32) - 128.0) / 128.0
        else:
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
        return _resample_if_needed(x, int(sr), 16000)
    except Exception:
        pass

    # 3) raw PCM16 little-endian
    try:
        x = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    except Exception:
        return np.zeros(1600, dtype=np.float32)  # 0.1s @ 16k


def wav_bytes_from_float32(x: np.ndarray, sr: int) -> bytes:
    """Write mono PCM16 WAV bytes from float32 waveform [-1,1]."""
    x16 = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()

    def w(u32): buf.write(struct.pack("<I", u32))
    def h(u16): buf.write(struct.pack("<H", u16))

    data = x16.tobytes()
    buf.write(b"RIFF"); w(36 + len(data)); buf.write(b"WAVE")
    buf.write(b"fmt "); w(16); h(1); h(1); w(sr); w(sr*2); h(2); h(16)
    buf.write(b"data"); w(len(data)); buf.write(data)
    return buf.getvalue()
