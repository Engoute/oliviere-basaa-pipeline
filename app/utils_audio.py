import io, wave, numpy as np, torch, torchaudio
from .config import S

def pcm16le_bytes_to_float32_mono(pcm: bytes, sr=S.in_pcm_sr) -> torch.Tensor:
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.tensor(arr).unsqueeze(0)  # (1, n)
    return wav  # mono

def resample(wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr: return wav
    return torchaudio.functional.resample(wav, src_sr, dst_sr)

def wav_bytes_from_float32(wav: np.ndarray, sr: int) -> bytes:
    wav16 = np.clip(wav * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wav16.tobytes())
    return buf.getvalue()
