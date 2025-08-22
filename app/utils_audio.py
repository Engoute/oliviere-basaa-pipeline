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
