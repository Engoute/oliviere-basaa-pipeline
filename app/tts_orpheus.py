# FILE: app/tts_orpheus.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Orpheus:
    """
    Orpheus-3B acoustic model + SNAC vocoder.

    What changed vs. your previous version (based on our Colab findings):
      • Always resolve a DIRECTORY for the acoustic model (never a single file).
      • Force the FAST tokenizer (tokenizer.json requires it).
      • Provide attention_mask and pad_token_id to silence generation warnings.

    Expected layout:
      - Acoustic model (weights + tokenizer.json):
          /data/models/orpheus_3b_resolved  (symlink created by bootstrap)
        or a bundle root that contains an acoustic subdir with config.json + tokenizer.json.
      - Vocoder:
          <bundle_root>/vocoder/{config.json,pytorch_model.bin}

    Output:
      - float32 mono waveform at sr_out (default 24000).
    """

    BASE = 128_266
    LANE_OFFSETS = [0, 4096, 8192, 12_288, 16_384, 20_480, 24_576]

    # ------------------------- init -------------------------
    def __init__(self, bundle_root: str, sr_out: int = 24_000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.sr = int(sr_out)

        # 1) Resolve acoustic directory (never a single file)
        acoustic = self._resolve_acoustic_dir(bundle_root)
        assert (acoustic / "config.json").exists(), f"[tts] Orpheus config.json not found in {acoustic}"

        # 2) FAST tokenizer is REQUIRED for tokenizer.json-only bundles
        self.tok = AutoTokenizer.from_pretrained(
            str(acoustic),
            local_files_only=True,
            use_fast=True,              # <-- critical fix
            trust_remote_code=True,
        )

        # 3) Load acoustic LM
        self.lm = AutoModelForCausalLM.from_pretrained(
            str(acoustic),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # 4) SNAC vocoder from bundle_root/vocoder
        vocoder_dir = Path(bundle_root) / "vocoder"
        cfg = vocoder_dir / "config.json"
        st = vocoder_dir / "pytorch_model.bin"
        assert cfg.exists() and st.exists(), f"[tts] vocoder assets missing under {vocoder_dir}"

        with open(cfg, "r", encoding="utf-8") as f:
            vcfg = json.load(f)
        # Allow vocoder config to define sampling_rate
        self.sr = int(vcfg.get("sampling_rate", self.sr))

        from snac import SNAC
        self.voc = SNAC(**vcfg).to(self.device).eval()
        state = torch.load(st, map_location=self.device)
        # load only matching keys to be robust to tiny mismatches
        sd = self.voc.state_dict()
        ok = {k: v for k, v in state.items() if k in sd and v.shape == sd[k].shape}
        self.voc.load_state_dict(ok, strict=False)

        print(f"[tts] Orpheus ready (sr={self.sr} Hz)")

    # ------------------ path resolution helpers ------------------
    def _resolve_acoustic_dir(self, bundle_root: str) -> Path:
        """
        Prefer the bootstrap symlink /data/models/orpheus_3b_resolved if it exists.
        Otherwise, search the provided bundle_root for a folder that contains
        config.json + tokenizer.json (+ weights).
        Returns a directory path.
        """
        # Preferred symlink from bootstrap.py
        symlink = Path("/data/models/orpheus_3b_resolved")
        if symlink.exists() and symlink.is_dir():
            return symlink

        base = Path(bundle_root)
        # If bundle_root itself looks like the acoustic dir, use it
        if (base / "config.json").exists() and (base / "tokenizer.json").exists():
            return base

        # Otherwise, scan subdirectories for a proper acoustic folder
        best: Optional[Path] = None
        for cfg in base.rglob("config.json"):
            d = cfg.parent
            if (d / "tokenizer.json").exists():
                has_weights = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
                if has_weights:
                    # choose the shortest path (usually the right one)
                    if best is None or len(str(d)) < len(str(best)):
                        best = d
        return best or base  # last-resort: return base and let asserts catch it

    # ------------------ token decoding helpers -------------------
    def _is_audio_token(self, t: int, i: int) -> bool:
        lo = self.BASE + self.LANE_OFFSETS[i % 7]
        hi = lo + 4096 - 1
        return lo <= t <= hi

    def _tokens_to_codes(self, tail: List[int], frames_cap: Optional[int] = None):
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

    # --------------------------- TTS ---------------------------
    def tts(self, text_lg: str) -> np.ndarray:
        # Minimal prompt: voice selector + text + request audio tokens
        # (Keep your markers; we just ensure bos/eos availability.)
        bos = self.tok.bos_token or "<s>"
        eos = self.tok.eos_token or "</s>"
        prompt = f"{bos}<|voice|>basaa_speaker<|text|>{text_lg}{eos}<|audio|>"

        enc = self.tok(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,   # <-- give attention_mask to avoid warnings
        )
        in_ids = enc["input_ids"].to(self.lm.device)
        attn = enc["attention_mask"].to(self.lm.device)

        # Make sure pad_token_id is defined (avoid runtime warnings)
        pad_id = self.tok.eos_token_id
        if pad_id is None:
            pad_id = int(self.tok.convert_tokens_to_ids(eos))
        if getattr(self.lm.generation_config, "pad_token_id", None) is None:
            self.lm.generation_config.pad_token_id = pad_id
        if getattr(self.lm.config, "pad_token_id", None) is None:
            self.lm.config.pad_token_id = pad_id

        with torch.inference_mode():
            out = self.lm.generate(
                in_ids,
                attention_mask=attn,
                max_new_tokens=12_000,    # your original limit
                do_sample=False,
                pad_token_id=pad_id,      # explicit to silence warnings
                return_dict_in_generate=True,
            )

        tail = out.sequences[0][in_ids.shape[-1]:].tolist()
        codes, frames = self._tokens_to_codes(tail, frames_cap=4_000)
        if codes is None or frames == 0:
            return np.zeros(0, dtype=np.float32)

        with torch.inference_mode():
            wav = self.voc.decode(codes).detach().float().cpu().numpy().squeeze()

        return np.asarray(wav, dtype=np.float32).reshape(-1)
