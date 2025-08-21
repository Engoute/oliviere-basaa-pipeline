# FILE: app/tts_orpheus.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import S
from .utils_audio import resample as _resample

class Orpheus:
    # Token-to-code map assumptions (as per bundle spec)
    BASE = 128_266
    LANE_OFFSETS = [0, 4096, 8192, 12_288, 16_384, 20_480, 24_576]

    def __init__(self, path: str, sr_out: int = 24_000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        self.sr_out = sr_out

        # Acoustic model dir (bootstrap created /data/models/orpheus_3b_resolved)
        acoustic = Path("/data/models/orpheus_3b_resolved")
        assert (acoustic / "config.json").exists(), f"Orpheus acoustic config.json missing at {acoustic}"

        self.tok = AutoTokenizer.from_pretrained(acoustic, local_files_only=True, use_fast=False)
        self.lm  = AutoModelForCausalLM.from_pretrained(
            acoustic, local_files_only=True, trust_remote_code=True,
            torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()

        # Local SNAC vocoder (bundled)
        vocoder_dir = Path(S.path_orpheus) / "vocoder"
        cfg_path = vocoder_dir / "config.json"
        state_path = vocoder_dir / "pytorch_model.bin"
        assert cfg_path.exists() and state_path.exists(), f"Vocoder files missing under {vocoder_dir}"
        cfg = json.loads(cfg_path.read_text())

        from snac import SNAC
        self.voc = SNAC(**cfg).to(self.device).eval()
        state = torch.load(state_path, map_location=self.device)
        # be lenient with key/shape mismatches
        sd = self.voc.state_dict()
        filtered = {k: v for k, v in state.items() if k in sd and tuple(v.shape) == tuple(sd[k].shape)}
        self.voc.load_state_dict(filtered, strict=False)
        self.sr_vocoder = int(cfg.get("sampling_rate", 24000))
        print(f"[tts] Orpheus loaded. vocoder_sr={self.sr_vocoder} out_sr={self.sr_out}")

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
        valid  = valid[: frames * 7]
        if frames == 0:
            return None, 0
        if frames_cap is not None and frames > frames_cap:
            valid = valid[: frames_cap * 7]
            frames = frames_cap

        lvl0, lvl1, lvl2 = [], [], []
        for i in range(0, len(valid), 7):
            f0 = valid[i+0] - (self.BASE + self.LANE_OFFSETS[0])
            f1 = valid[i+1] - (self.BASE + self.LANE_OFFSETS[1])
            f2 = valid[i+2] - (self.BASE + self.LANE_OFFSETS[2])
            f3 = valid[i+3] - (self.BASE + self.LANE_OFFSETS[3])
            f4 = valid[i+4] - (self.BASE + self.LANE_OFFSETS[4])
            f5 = valid[i+5] - (self.BASE + self.LANE_OFFSETS[5])
            f6 = valid[i+6] - (self.BASE + self.LANE_OFFSETS[6])
            lvl0.append(f0); lvl1.extend([f1, f4]); lvl2.extend([f2, f3, f5, f6])

        codes = [
            torch.tensor(lvl0, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.tensor(lvl1, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.tensor(lvl2, dtype=torch.long, device=self.device).unsqueeze(0),
        ]
        return codes, frames

    def tts(self, text_lg: str) -> np.ndarray:
        # Prompt format may vary per bundle; this follows the provided spec.
        prompt = f"{self.tok.bos_token or '<s>'}<|voice|>basaa_speaker<|text|>{text_lg}{self.tok.eos_token or '</s>'}<|audio|>"
        in_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            out = self.lm.generate(
                in_ids, max_new_tokens=12000, do_sample=False, return_dict_in_generate=True, early_stopping=True
            )
        tail = out.sequences[0][in_ids.shape[-1]:].tolist()
        codes, frames = self._tokens_to_codes(tail, frames_cap=4000)
        if codes is None:
            return np.zeros(0, dtype=np.float32)

        with torch.inference_mode():
            wav = self.voc.decode(codes).detach().float().cpu().numpy().squeeze()

        # Resample to requested output SR if needed
        if self.sr_vocoder != self.sr_out:
            wav = _resample(wav, self.sr_vocoder, self.sr_out)
        return wav.astype(np.float32).reshape(-1)
