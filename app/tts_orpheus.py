import json, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC
from .config import S

# This mirrors your “SafeStop + Trim” logic but returns a single WAV at 24 kHz.
class Orpheus:
    def __init__(self, path: str, sr_out: int = 24000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.lm  = AutoModelForCausalLM.from_pretrained(path, torch_dtype=self.dtype, low_cpu_mem_usage=True).to(self.device).eval()

        # SNAC vocoder (we assume it is cached when used first time)
        # If you already have the SNAC snapshot cached, we can point to it; otherwise it will download on first use.
        # For offline-only deployments, bake SNAC into a separate bundle if you want.
        # Here we load the default 24k config that SNAC exposes.
        cfg_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--hubertsiuzdak--snac_24khz" / "snapshots"
        # pick the newest snapshot folder
        snap = sorted(cfg_path.glob("*/config.json"))[-1]
        cfg = json.loads(Path(snap).read_text())
        self.sr = int(cfg.get("sampling_rate", 24000))
        state_path = Path(str(snap).replace("config.json", "pytorch_model.bin"))
        self.voc = SNAC(**cfg).to(self.device).eval()
        state = torch.load(state_path, map_location=self.device)
        state_ok = {k:v for k,v in state.items() if k in self.voc.state_dict() and v.shape == self.voc.state_dict()[k].shape}
        self.voc.load_state_dict(state_ok, strict=False)

        # audio token mapping
        self.BASE = 128_266
        self.LANE_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]

    def _is_audio_token(self, t, i):
        lo = self.BASE + self.LANE_OFFSETS[i % 7]
        hi = lo + 4096 - 1
        return lo <= t <= hi

    def _tokens_to_codes(self, tail, frames_cap=None):
        valid = []
        for i, t in enumerate(tail):
            if self._is_audio_token(t, i): valid.append(t)
            else: break
        frames = len(valid) // 7
        valid  = valid[: frames * 7]
        if frames == 0: return None, 0
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
        codes = [torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0) for x in (lvl0, lvl1, lvl2)]
        return codes, frames

    def tts(self, text_lg: str) -> np.ndarray:
        # Conservative chunking; single-pass return
        prompt = f"{self.tok.bos_token or '<s>'}<|voice|>basaa_speaker<|text|>{text_lg}{self.tok.eos_token or '</s>'}<|audio|>"
        in_ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            out = self.lm.generate(
                in_ids,
                max_new_tokens=12000,  # ample budget; clipped below
                do_sample=False,
                return_dict_in_generate=True,
            )
        tail = out.sequences[0][in_ids.shape[-1]:].tolist()
        codes, frames = self._tokens_to_codes(tail, frames_cap=4000)
        if codes is None: return np.zeros(0, dtype=np.float32)
        with torch.inference_mode():
            wav = self.voc.decode(codes).detach().float().cpu().numpy().squeeze()
        return wav  # 24k float32
