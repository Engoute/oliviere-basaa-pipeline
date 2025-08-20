from faster_whisper import WhisperModel
from .config import S

class ASR:
    def __init__(self, ct2_path: str):
        # Runs on GPU by default; fall back to CPU automatically if needed.
        self.model = WhisperModel(ct2_path, device="cuda", compute_type="float16")

    def transcribe(self, wav16k: bytes):
        # wav16k: raw PCM16LE @16kHz (already from client)
        # faster-whisper accepts (samples, sr) but we’ll pass bytes via vad filter
        segments, info = self.model.transcribe(
            wav16k, language=None, beam_size=1, vad_filter=True, temperature=0.0
        )
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        lang = (info.language or "unk").lower()
        return text, lang, info.language_probability or 0.0
