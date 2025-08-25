# FILE: app/utils_audio.py
import io, struct, numpy as np

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

def decode_audio_to_16k_float_mono(blob: bytes) -> np.ndarray:
    """
    Robust decode for either WAV or raw PCM16. Uses torchaudio to sniff+decode,
    then resamples to 16 kHz mono float32.
    """
    import torch, torchaudio
    try:
        # Try as a WAV/MP3/etc.
        wav, sr = torchaudio.load(io.BytesIO(blob))
    except Exception:
        # Fallback: assume little-endian PCM16 mono @ 48k
        arr = np.frombuffer(blob, dtype="<i2").astype(np.float32) / 32768.0
        wav = torch.tensor(arr).unsqueeze(0)  # [1,T]
        sr = 48000

    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0).contiguous().cpu().numpy().astype(np.float32)
