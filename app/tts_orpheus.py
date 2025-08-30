# FILE: app/tts_orpheus.py
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Optional, List, Tuple, Callable, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Avoid tokenizer parallelism spam
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# SNAC vocoder
from snac import SNAC


def _to_channels_time(x: torch.Tensor) -> torch.Tensor:
    """Ensure [C, T] for writing/returning. Flattens any extra dims into channels."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


class Orpheus:
    """
    Orpheus acoustic model + SNAC vocoder.

    Bundle layout expected:
      <bundle_root>/
        acoustic_model/   (HF model dir: config.json, tokenizer.json, safetensors…)
        vocoder/          (SNAC dir: config.json + weights)

    Public:
      - tts(text_lg: str, max_new_audio_tokens: Optional[int] = None) -> np.ndarray (float32 mono @ sr_out)
      - iter_pcm_stream(text_lg, fmt="pcm16le", chunk_ms=120): yields raw PCM bytes
    """

    DEFAULT_AUDIO_CODE_OFFSET = 128_266  # 7-lane packing offset you’ve been using

    def __init__(
        self,
        bundle_root: str,
        sr_out: int = 24_000,
        voice_name: str = "basaa_speaker",
        audio_code_offset: int = DEFAULT_AUDIO_CODE_OFFSET,
        min_frames: int = 24,
        temperature: float = 0.65,
        top_p: float = 0.94,
        repetition_penalty: float = 1.12,
        device: Optional[str] = None,
        tail_pad_ms: int = 60,  # ← tiny safety pad to avoid cut-offs on playback
    ):
        self.sr_out = int(sr_out)
        self.voice_name = voice_name
        self.code_offset = int(audio_code_offset)
        self.min_frames = int(min_frames)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.tail_pad_ms = int(max(0, tail_pad_ms))

        # Device + dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # Resolve acoustic + vocoder dirs
        bundle_root = os.path.abspath(bundle_root)
        acoustic_dir = self._resolve_acoustic_dir(bundle_root)
        if not (acoustic_dir and Path(acoustic_dir).is_dir()):
            raise ValueError(f"[tts] acoustic path must be a directory with tokenizer+weights. Got: {acoustic_dir}")
        if not (Path(acoustic_dir) / "tokenizer.json").exists():
            raise FileNotFoundError(f"[tts] tokenizer.json not found in {acoustic_dir}")
        if not ((Path(acoustic_dir) / "model.safetensors").exists() or list(Path(acoustic_dir).glob("model-*.safetensors"))):
            raise FileNotFoundError(f"[tts] model safetensors not found in {acoustic_dir}")
        print(f"[tts] acoustic dir = {acoustic_dir}")

        vocoder_dir = self._find_vocoder_dir(bundle_root)
        if vocoder_dir is None:
            raise FileNotFoundError(
                f"[tts] vocoder assets not found under bundle root: {bundle_root}\n"
                f"Expected a folder like .../vocoder/ with config.json + weights."
            )
        print(f"[tts] vocoder dir = {vocoder_dir}")

        # Load tokenizer/model
        self.tok = AutoTokenizer.from_pretrained(
            acoustic_dir,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            acoustic_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        # SNAC vocoder
        self.snac = SNAC.from_pretrained(vocoder_dir).to(self.device).eval()

        # Derive FPS & sampling rate from vocoder config when present
        try:
            with open(os.path.join(vocoder_dir, "config.json"), "r", encoding="utf-8") as f:
                vcfg = json.load(f)
            sr = int(vcfg.get("sampling_rate", self.sr_out))
            hop = vcfg.get("hop_length")
            self.frames_per_sec = (sr / hop) if hop else 75.0
            self.sr_out = sr  # honor vocoder sr
        except Exception:
            self.frames_per_sec = 75.0

        print(f"[tts] Orpheus ready (sr={self.sr_out} Hz, device={self.device})")

    # ---------- robust, non-streaming TTS ----------
    @torch.no_grad()
    def tts(self, basaa_text: str, max_new_audio_tokens: Optional[int] = None) -> np.ndarray:
        """
        Generate full waveform (float32 mono, [-1..1], sr_out).
        Adds a tiny tail pad to reduce risk of client-side cutoff.
        """
        prompt = self._format_prompt(basaa_text)
        enc = self.tok(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        max_new = int(max_new_audio_tokens or self._estimate_max_tokens_from_text(basaa_text))

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            repetition_penalty=float(self.repetition_penalty),
            max_new_tokens=max_new,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )

        tail = gen[0, input_ids.shape[-1]:].tolist()
        while tail and self.tok.eos_token_id is not None and tail[-1] == self.tok.eos_token_id:
            tail.pop()

        if not tail:
            return np.zeros((self.sr_out // 20,), dtype=np.float32)

        codes, _ = self._decode_llm_tokens_to_snac_codes(tail, self.code_offset, self.min_frames)
        wav = self.snac.decode(codes)  # [1, T] or similar
        wav = torch.clamp(wav, -1, 1).contiguous()
        wav = _to_channels_time(wav).to(torch.float32)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.shape[1] == 0:
            return np.zeros((self.sr_out // 20,), dtype=np.float32)
        out = wav.squeeze(0).cpu().numpy().astype(np.float32)

        # Small fade-out + safety pad at tail to prevent click/cut-off
        out = self._apply_tail_safety_pad(out)
        return out

    # ---------- streaming by slicing the final WAV ----------
    def iter_pcm_stream(
        self,
        text_lg: str,
        fmt: str = "pcm16le",   # "pcm16le" or "f32le"
        chunk_ms: int = 120,
        max_new_tokens: Optional[int] = None,
    ) -> Iterable[bytes]:
        """
        Non-token streaming: synthesize once (same audio as non-streaming),
        then slice to chunks in the requested PCM format.
        """
        wav = self.tts(text_lg, max_new_audio_tokens=max_new_tokens)
        if wav.size == 0:
            return
        step = max(1, int(self.sr_out * (chunk_ms / 1000.0)))
        if fmt.lower() == "f32le":
            wbytes = wav.astype("<f4", copy=False).tobytes(order="C")
            for i in range(0, len(wbytes), step * 4):
                yield wbytes[i:i + step * 4]
        elif fmt.lower() == "pcm16le":
            x16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
            wbytes = x16.tobytes(order="C")
            for i in range(0, len(wbytes), step * 2):
                yield wbytes[i:i + step * 2]
        else:
            raise ValueError(f"Unsupported stream format: {fmt}")

    # ---------- internals ----------
    def _format_prompt(self, text: str) -> str:
        bos = self.tok.bos_token or ""
        eos = self.tok.eos_token or ""
        return f"{bos}<|voice|>{self.voice_name}<|text|>{text}{eos}<|audio|>"

    def _estimate_max_tokens_from_text(self, text: str) -> int:
        # Slightly more generous estimate + fixed guard band to avoid short audio
        n_words = max(1, len(re.findall(r"\w+", text)))
        rough = n_words * 38 + 56
        return int(max(9 * self.min_frames, min(4 * 4096, rough)))

    def _apply_tail_safety_pad(self, wav: np.ndarray, pad_ms: Optional[int] = None) -> np.ndarray:
        pad_ms = self.tail_pad_ms if pad_ms is None else int(max(0, pad_ms))
        if pad_ms <= 0 or wav.size == 0:
            return wav
        # short fade-out to zero over ~5 ms (avoid clicks)
        fade = int(min(max(1, self.sr_out // 200), wav.size))  # ~5ms
        if fade > 0:
            ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            wav[-fade:] *= ramp
        n = int(self.sr_out * (pad_ms / 1000.0))
        if n <= 0:
            return wav
        pad = np.zeros(n, dtype=np.float32)
        return np.concatenate([wav, pad], dtype=np.float32)

    @staticmethod
    def _decode_llm_tokens_to_snac_codes(
        ids: List[int],
        code_offset: int,
        need_min_frames: int = 24,
    ) -> Tuple[List[torch.Tensor], int]:
        raw = [t - code_offset - ((i % 7) * 4096) for i, t in enumerate(ids)]
        n_frames = len(raw) // 7
        raw = raw[: n_frames * 7]

        C, M, F = [], [], []
        valid = 0
        for i in range(n_frames):
            a = i * 7
            frame = raw[a:a + 7]
            if any((x < 0 or x >= 4096) for x in frame):
                continue
            c0, m0, f0, f1, m1, f2, f3 = frame
            C.append(c0)
            M.extend([m0, m1])
            F.extend([f0, f1, f2, f3])
            valid += 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C = torch.tensor(C, dtype=torch.long, device=device).unsqueeze(0)
        M = torch.tensor(M, dtype=torch.long, device=device).unsqueeze(0)
        F = torch.tensor(F, dtype=torch.long, device=device).unsqueeze(0)
        return [C, M, F], valid

    @staticmethod
    def _resolve_acoustic_dir(bundle_root: str) -> Optional[str]:
        root = Path(bundle_root)
        if (root / "acoustic_model").is_dir():
            return str((root / "acoustic_model").resolve())
        if (root / "config.json").exists() and (root / "tokenizer.json").exists():
            return str(root.resolve())
        best: Optional[Path] = None
        for cfg in root.rglob("config.json"):
            d = cfg.parent
            if (d / "tokenizer.json").exists() and (
                (d / "model.safetensors").exists() or list(d.glob("model-*.safetensors"))
            ):
                if best is None or len(str(d)) < len(str(best)):
                    best = d
        return str(best.resolve()) if best else None

    @staticmethod
    def _find_vocoder_dir(bundle_root: str) -> Optional[str]:
        root = Path(bundle_root)
        vd = root / "vocoder"
        if vd.is_dir() and (vd / "config.json").exists():
            return str(vd.resolve())
        for cfg in root.rglob("config.json"):
            d = cfg.parent
            if "vocoder" in d.name.lower():
                return str(d.resolve())
        return None
