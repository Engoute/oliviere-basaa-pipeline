# FILE: app/tts_orpheus.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Orpheus:
    """
    Orpheus-3B acoustic model + SNAC vocoder.
    - Expects acoustic model at /data/models/orpheus_3b_resolved (symlink set by bootstrap)
    - Expects vocoder under PATH_ORPHEUS/vocoder/{config.json,pytorch_model.bin}
    - Outputs float32 mono at sr_out (default 24000).
    """
    BASE = 128_266
    LANE_OFFSETS = [0, 4096, 8192, 12_288, 16_384, 20_480, 24_576]

    def __init__(self, bundle_root: str, sr_out: int = 24_000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        self.sr = int(sr_out)

        # Acoustic model (bootstrap created this symlink)
        acoustic = Path("/data/models/orpheus_3b_resolved")
        if not (acoustic / "config.json").exists():
            # Fallback: maybe acoustic is directly the bundle root
            acoustic = Path(bundle_root)
        assert (acoustic / "config.json").exists(), f"[tts] Orpheus config.json not found in {acoustic}"

        self.tok = AutoTokenizer.from_pretrained(acoustic, local_files_only=True, use_fast=False)
        self.lm  = AutoModelForCausalLM.from_pretrained(
            acoustic, local_files_only=True, trust_remote_code=True,
            torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()

        # Vocoder
        vocoder_dir = Path(bundle_root) / "vocoder"
        cfg = (vocoder_dir / "config.json")
        st  = (vocoder_dir / "pytorch_model.bin")
        assert cfg.exists() and st.exists(), f"[tts] vocoder assets missing under {vocoder_dir}"
        import json
        with open(cfg, "r") as f:
            vcfg = json.load(f)
        self.sr = int(vcfg.get("sampling_rate", self.sr))

        from snac import SNAC
        self.voc = SNAC(**vcfg).to(self.device).eval()
        state = torch.load(st, map_location=self.device)
        # allow partial strictness
        sd = self.voc.state_dict()
        ok = {k: v for k, v in state.items() if k in sd and v.shape == sd[k].shape}
        self.voc.load_state_dict(ok, strict=False)
        print(f"[tts] Orpheus ready (sr={self.sr} Hz)")

    # --- decode helpers ---
    def _is_audio_token(self, t: int, i: int) -> bool:
        lo = self.BASE + self.LANE_OFFSETS[i % 7]
        hi = lo + 4096 - 1
        return lo <= t <= hi

    def _tokens_to_codes(self, tail: List[int], frames_cap: Optional[int] = None):
        valid = []
        for i, t in enumerate(tail):
            if self._is_audio_token(t, i): valid.append(t)
            else: break
        frames = len(valid) // 7
        valid  = valid[: frames * 7]
        if frames == 0: return None, 0
        if frames_cap is not None and frames > frames_cap:
            valid = valid[: frames_cap * 7]; frames = frames_cap

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
        codes = [torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0) for x in (lvl0, lvl1, lvl2)]
        return codes, frames

    def tts(self, text_lg: str) -> np.ndarray:
        # Minimal prompt: voice selector + text + request audio tokens
        prompt = f"{self.tok.bos_token or '<s>'}<|voice|>basaa_speaker<|text|>{text_lg}{self.tok.eos_token or '</s>'}<|audio|>"
        in_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.lm.device)

        with torch.inference_mode():
            out = self.lm.generate(
                in_ids,
                max_new_tokens=12000,
                do_sample=False,
                return_dict_in_generate=True,
            )
        tail = out.sequences[0][in_ids.shape[-1]:].tolist()
        codes, frames = self._tokens_to_codes(tail, frames_cap=4000)
        if codes is None or frames == 0:
            return np.zeros(0, dtype=np.float32)

        with torch.inference_mode():
            wav = self.voc.decode(codes).detach().float().cpu().numpy().squeeze()
        return np.asarray(wav, dtype=np.float32).reshape(-1)
