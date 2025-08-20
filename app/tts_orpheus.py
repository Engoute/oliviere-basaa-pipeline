import json, torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC
from .config import S

RESOLVED_ORPHEUS = Path("/data/models/orpehus_3b_resolved")  # set by bootstrap

class Orpheus:
    def __init__(self, path: str, sr_out: int = 24000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32

        # prefer resolved path if available / valid
        p = Path(path)
        use = RESOLVED_ORPHEUS if (RESOLVED_ORPHEUS / "config.json").exists() else p
        if not (use / "config.json").exists():
            # last chance: search inside the provided path
            cand = next((c.parent for c in use.rglob("config.json")), use)
            use = cand

        # Tokenizer / model – allow custom architecture
        self.tok = AutoTokenizer.from_pretrained(
            use, local_files_only=True, use_fast=False
        )
        self.lm  = AutoModelForCausalLM.from_pretrained(
            use, local_files_only=True, trust_remote_code=True,
            torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()

        # --- SNAC vocoder (same as before) ---
        cfg_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--hubertsiuzdak--snac_24khz" / "snapshots"
        snaps = sorted(cfg_root.glob("*/config.json"))
        if not snaps:
            raise FileNotFoundError("SNAC 24 kHz not cached yet. Run once online or bake into image.")
        cfg_path = snaps[-1]
        cfg = json.loads(cfg_path.read_text())
        self.sr = int(cfg.get("sampling_rate", 24000))
        state_path = Path(str(cfg_path).replace("config.json", "pytorch_model.bin"))

        self.voc = SNAC(**cfg).to(self.device).eval()
        state = torch.load(state_path, map_location=self.device)
        state_ok = {k: v for k, v in state.items() if k in self.voc.state_dict() and v.shape == self.voc.state_dict()[k].shape}
        self.voc.load_state_dict(state_ok, strict=False)

        # audio token mapping (unchanged)
        self.BASE = 128_266
        self.LANE_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]

    # ... rest of your class unchanged ...
