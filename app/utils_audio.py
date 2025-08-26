# FILE: app/utils_audio.py
import io, struct
import numpy as np

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

def _pcm16le_bytes_to_float_mono(raw: bytes, sr: int = 16000) -> np.ndarray:
    # Assume raw little-endian PCM16 mono at sr
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return arr

def decode_audio_to_16k_float_mono(data: bytes) -> np.ndarray:
    """
    Best-effort decode of arbitrary byte blob to 16k mono float32.
    Tries torchaudio (WAV/MP3/etc). Falls back to raw PCM16 LE at 16k.
    """
    try:
        import torchaudio, torch, io as _io
        with _io.BytesIO(data) as bio:
            wav, srate = torchaudio.load(bio)  # [C, T], float32 or int16
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
            if wav.abs().max() > 0:
                wav = wav / (wav.abs().max() + 1e-8)
        # mono
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # resample
        if int(srate) != 16000:
            wav = torchaudio.functional.resample(wav, srate, 16000)
        wav = wav.squeeze(0).cpu().numpy().astype(np.float32)
        return wav
    except Exception:
        # Fallback: raw PCM16 mono @ 16k
        try:
            return _pcm16le_bytes_to_float_mono(data, 16000)
        except Exception:
            return np.zeros((0,), dtype=np.float32)
