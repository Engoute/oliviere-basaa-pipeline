# FILE: app/tts_orpheus.py
from __future__ import annotations

import os
import math
import json
from pathlib import Path
from typing import Optional, List, Tuple, Callable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Avoid tokenizer parallelism spam; safe default for services
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# SNAC vocoder (works with local folder via from_pretrained)
from snac import SNAC


def _to_channels_time(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure [C, T] for writing/returning. Flattens any extra dims into channels.
    """
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


class Orpheus:
    """
    Orpheus acoustic model + SNAC vocoder (non-streaming, robust).

    Bundle layout expected:
      <bundle_root>/
        acoustic_model/   (HF model dir: config.json, tokenizer.json, safetensors…)
        vocoder/          (SNAC dir: config.json + weights)

    Public:
      - tts(text_lg: str, max_new_audio_tokens: Optional[int] = None) -> np.ndarray
      - stream_tts(text_lg, on_chunk, chunk_ms=600): non-streaming "chunker"
        (generates once, then slices PCM into callback chunks)
    """

    # Audio token packing: 7 lanes per frame, each lane in 0..4095
    # Single global offset used during finetune (matches your working runs)
    DEFAULT_AUDIO_CODE_OFFSET = 128_266

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
    ):
        self.sr_out = int(sr_out)
        self.voice_name = voice_name
        self.code_offset = int(audio_code_offset)
        self.min_frames = int(min_frames)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)

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

        # Load SNAC vocoder (auto-loads config + weights from folder)
        self.snac = SNAC.from_pretrained(vocoder_dir).to(self.device).eval()

        # Derive frames-per-second if present in config, else ~75 fps
        try:
            with open(os.path.join(vocoder_dir, "config.json"), "r", encoding="utf-8") as f:
                vcfg = json.load(f)
            sr = int(vcfg.get("sampling_rate", self.sr_out))
            hop = vcfg.get("hop_length")
            self.frames_per_sec = (sr / hop) if hop else 75.0
            # honor vocoder sampling rate for synthesis if present
            self.sr_out = sr
        except Exception:
            self.frames_per_sec = 75.0

        print(f"[tts] Orpheus ready (sr={self.sr_out} Hz, device={self.device})")

    # ---------- public: robust, non-streaming TTS ----------
    @torch.no_grad()
    def tts(self, basaa_text: str, max_new_audio_tokens: Optional[int] = None) -> np.ndarray:
        """
        Generate full waveform (float32 mono, [-1..1], sr_out).
        """
        prompt = self._format_prompt(basaa_text)
        enc = self.tok(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        # Token budget from text length (same heuristic you validated in Colab).
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
            eos_token_id=self.tok.eos_token_id,  # text EOS; audio ends when codes stop or budget ends
        )

        # Extract only new tokens
        tail = gen[0, input_ids.shape[-1]:].tolist()
        # Trim trailing text EOS if any
        while tail and self.tok.eos_token_id is not None and tail[-1] == self.tok.eos_token_id:
            tail.pop()

        if not tail:
            # return a tiny silence instead of empty, to keep clients happy
            return np.zeros((self.sr_out // 20,), dtype=np.float32)

        # Convert to SNAC code streams
        codes, n_valid = self._decode_llm_tokens_to_snac_codes(tail, self.code_offset, self.min_frames)
        # Decode to waveform
        wav = self.snac.decode(codes)  # usually [1, T] (or [1,1,T])
        wav = torch.clamp(wav, -1, 1).contiguous()
        wav = _to_channels_time(wav).to(torch.float32)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono
        if wav.shape[1] == 0:
            return np.zeros((self.sr_out // 20,), dtype=np.float32)

        return wav.squeeze(0).cpu().numpy().astype(np.float32)

    # ---------- optional: "streaming" by slicing the final WAV ----------
    def stream_tts(
        self,
        text_lg: str,
        on_chunk: Callable[[np.ndarray], None],
        chunk_ms: int = 600,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Non-streaming generation under the hood. We synthesize once, then slice
        the waveform into ~chunk_ms chunks and invoke on_chunk for each.
        This avoids token-level streaming & WS race conditions.
        """
        wav = self.tts(text_lg, max_new_audio_tokens=max_new_tokens)
        if wav.size == 0:
            return
        step = max(1, int(self.sr_out * (chunk_ms / 1000.0)))
        for i in range(0, len(wav), step):
            chunk = wav[i:i + step]
            if chunk.size:
                on_chunk(chunk.astype(np.float32, copy=False))

    # ---------- internals ----------
    def _format_prompt(self, text: str) -> str:
        bos = self.tok.bos_token or ""
        eos = self.tok.eos_token or ""
        # Matches your successful prompt: <|voice|> + <|text|> … <|audio|>
        return f"{bos}<|voice|>{self.voice_name}<|text|>{text}{eos}<|audio|>"

    def _estimate_max_tokens_from_text(self, text: str) -> int:
        """
        Empirical budget from words (≈33 tokens/word), with floors/ceil.
        Ensures enough room but avoids runaway generations.
        """
        n_words = max(1, len(text.split()))
        rough = n_words * 33
        # Keep within sane bounds; also ensure we allow at least min_frames
        return int(max(7 * self.min_frames, min(4 * 4096, rough)))

    @staticmethod
    def _decode_llm_tokens_to_snac_codes(
        ids: List[int],
        code_offset: int,
        need_min_frames: int = 24,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Orpheus audio packing (7 tokens per frame):
          frame = [c0, m0, f0, f1, m1, f2, f3]
          LLM tokens = offset + [c0 + 0*4096, m0 + 1*4096, f0 + 2*4096, ... f3 + 6*4096]

        We de-interleave back into 3 SNAC streams (coarse, mid, fine),
        skipping any frame where a lane goes out of [0..4095].
        """
        raw = [t - code_offset - ((i % 7) * 4096) for i, t in enumerate(ids)]
        n_frames = len(raw) // 7
        raw = raw[: n_frames * 7]

        C, M, F = [], [], []
        valid = 0
        for i in range(n_frames):
            a = i * 7
            frame = raw[a:a + 7]
            if any((x < 0 or x >= 4096) for x in frame):
                continue  # skip invalid frame
            c0, m0, f0, f1, m1, f2, f3 = frame
            C.append(c0)
            M.extend([m0, m1])
            F.extend([f0, f1, f2, f3])
            valid += 1

        if valid < need_min_frames:
            # Let caller decide; we still return what we have to avoid hard failure
            pass

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        C = torch.tensor(C, dtype=torch.long, device=device).unsqueeze(0)
        M = torch.tensor(M, dtype=torch.long, device=device).unsqueeze(0)
        F = torch.tensor(F, dtype=torch.long, device=device).unsqueeze(0)
        return [C, M, F], valid

    # ---------- path helpers ----------
    @staticmethod
    def _resolve_acoustic_dir(bundle_root: str) -> Optional[str]:
        """
        Accept either:
          - .../acoustic_model/
          - bundle root that contains acoustic_model/
          - a directory that itself is a valid HF model dir
        """
        root = Path(bundle_root)
        if (root / "acoustic_model").is_dir():
            return str((root / "acoustic_model").resolve())

        # Direct "this dir is the model" layout
        if (root / "config.json").exists() and (root / "tokenizer.json").exists():
            return str(root.resolve())

        # Otherwise search shallowly for a model directory
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
        # Prefer explicit "vocoder" folder
        vd = root / "vocoder"
        if vd.is_dir() and (vd / "config.json").exists():
            return str(vd.resolve())

        # Fallback: search for a folder that looks like a SNAC repo dir
        for cfg in root.rglob("config.json"):
            d = cfg.parent
            if "vocoder" in d.name.lower():
                return str(d.resolve())
        return None
