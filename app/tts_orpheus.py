# app/tts_orpheus.py
"""
Orpheus-3B TTS (+ SNAC 24 kHz vocoder)

- Accepts your local bundle unzipped under /data/models/orpheus_3b (any nesting).
- bootstrap.py should create a stable symlink:
      /data/models/orpheus_3b_resolved -> folder that contains config.json
- Loads tokenizer/model with trust_remote_code=True (required by many TTS checkpoints).
- Finds the SNAC vocoder from the HF cache. If it's not cached yet, raises with
  a clear message (you can prefetch it once online or bake it into the image).

Output: np.ndarray dtype=float32 at 24 kHz (or the sampling_rate found in SNAC config).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC

# Optional project config (S.path_orpheus, etc.). Fallback to env if missing.
try:
    from .config import S  # type: ignore
except Exception:  # pragma: no cover
    class _S:
        path_orpheus = os.environ.get("PATH_ORPHEUS", "/data/models/orpheus_3b")
    S = _S()  # type: ignore


# -------- helpers --------

def _resolve_model_dir(base: Path) -> Path:
    """
    Return a directory that contains a config.json.
    First prefer '/data/models/orpheus_3b_resolved' (stable symlink created by bootstrap).
    Otherwise search inside 'base'.
    """
    stable = Path("/data/models/orpheus_3b_resolved")
    if (stable / "config.json").exists():
        return stable

    if (base / "config.json").exists():
        return base

    matches = list(base.rglob("config.json"))
    return matches[0].parent if matches else base


def _find_snac_snapshot() -> Tuple[Path, dict]:
    """
    Locate SNAC 24k vocoder in HF cache:
      ~/.cache/huggingface/hub/models--hubertsiuzdak--snac_24khz/snapshots/<hash>/
    Returns (snapshot_dir, cfg_dict). Raises with a clear message if not found.
    """
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--hubertsiuzdak--snac_24khz" / "snapshots"
    cfg_paths = sorted(root.glob("*/config.json"))
    if not cfg_paths:
        raise FileNotFoundError(
            "SNAC 24 kHz not found in HF cache.\n"
            "→ Run once with internet to cache it, or bake it into the image.\n"
            "   Repo: hubertsiuzdak/snac_24khz"
        )
    cfg_path = cfg_paths[-1]
    cfg = json.loads(cfg_path.read_text())
    return cfg_path.parent, cfg


# -------- main class --------

class Orpheus:
    """
    Simple one-shot TTS (no chunker/streamer here).
    Usage:
        tts = Orpheus(S.path_orpheus)          # or Orpheus(None) to rely on defaults
        wav = tts.tts("…Basaa text…")          # float32 mono at self.sr (24k)
    """

    # audio-token mapping constants used by the acoustic model
    BASE = 128_266
    LANE_OFFSETS = [0, 4096, 8192, 12_288, 16_384, 20_480, 24_576]

    def __init__(self, path: Optional[str] = None, sr_out: int = 24_000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Resolve acoustic model directory
        base = Path(path or getattr(S, "path_orpheus", "/data/models/orpheus_3b"))
        use = _resolve_model_dir(base)
        assert (use / "config.json").exists(), f"Orpheus config.json not found under: {use}"

        # Tokenizer + acoustic model (allow custom architectures)
        self.tok = AutoTokenizer.from_pretrained(
            use,
            local_files_only=True,
            use_fast=False,
        )
        self.lm = AutoModelForCausalLM.from_pretrained(
            use,
            local_files_only=True,
            trust_remote_code=True,     # IMPORTANT
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # SNAC 24 kHz vocoder
        snap_dir, cfg = _find_snac_snapshot()
        self.sr = int(cfg.get("sampling_rate", sr_out))
        self.voc = SNAC(**cfg).to(self.device).eval()

        # Load vocoder weights
        state_path = snap_dir / "pytorch_model.bin"
        state = torch.load(state_path, map_location=self.device)
        state_ok = {k: v for k, v in state.items()
                    if k in self.voc.state_dict() and v.shape == self.voc.state_dict()[k].shape}
        self.voc.load_state_dict(state_ok, strict=False)

    # ---- internals ----

    def _is_audio_token(self, t: int, i: int) -> bool:
        lo = self.BASE + self.LANE_OFFSETS[i % 7]
        hi = lo + 4096 - 1
        return lo <= t <= hi

    def _tokens_to_codes(self, tail: list[int], frames_cap: Optional[int] = None):
        """
        Extract contiguous audio tokens from the generated tail and map them into SNAC code streams.
        Returns (codes, frames) or (None, 0) if nothing valid was produced.
        """
        valid = []
        for i, t in enumerate(tail):
            if self._is_audio_token(t, i):
                valid.append(t)
            else:
                break

        frames = len(valid) // 7
        valid = valid[: frames * 7]
        if frames == 0:
            return None, 0

        if frames_cap is not None and frames > frames_cap:
            valid = valid[: frames_cap * 7]
            frames = frames_cap

        lvl0, lvl1, lvl2 = [], [], []
        for i in range(0, len(valid), 7):
            f0 = valid[i + 0] - (self.BASE + self.LANE_OFFSETS[0])
            f1 = valid[i + 1] - (self.BASE + self.LANE_OFFSETS[1])
            f2 = valid[i + 2] - (self.BASE + self.LANE_OFFSETS[2])
            f3 = valid[i + 3] - (self.BASE + self.LANE_OFFSETS[3])
            f4 = valid[i + 4] - (self.BASE + self.LANE_OFFSETS[4])
            f5 = valid[i + 5] - (self.BASE + self.LANE_OFFSETS[5])
            f6 = valid[i + 6] - (self.BASE + self.LANE_OFFSETS[6])
            lvl0.append(f0)
            lvl1.extend([f1, f4])
            lvl2.extend([f2, f3, f5, f6])

        codes = [
            torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0)
            for x in (lvl0, lvl1, lvl2)
        ]
        return codes, frames

    # ---- public API ----

    def tts(self, text_lg: str) -> np.ndarray:
        """
        Synthesize a single utterance.
        Returns: float32 mono waveform at self.sr (usually 24 kHz).
        """
        prompt = (
            f"{self.tok.bos_token or '<s>'}"
            f"<|voice|>basaa_speaker<|text|>{text_lg}"
            f"{self.tok.eos_token or '</s>'}<|audio|>"
        )
        in_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            out = self.lm.generate(
                in_ids,
                max_new_tokens=12_000,      # large budget; we'll cap after decoding tokens
                do_sample=False,
                return_dict_in_generate=True,
            )

        # take only the generated suffix
        tail = out.sequences[0][in_ids.shape[-1]:].tolist()

        # map to vocoder codes (apply conservative cap)
        codes, frames = self._tokens_to_codes(tail, frames_cap=4000)
        if codes is None:
            return np.zeros(0, dtype=np.float32)

        with torch.inference_mode():
            wav = self.voc.decode(codes).detach().float().cpu().numpy().squeeze()

        # ensure float32 mono ndarray
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav
