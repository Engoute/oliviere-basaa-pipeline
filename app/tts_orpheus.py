# FILE: app/tts_orpheus.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Callable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

# Avoid tokenizer parallelism spam; safe default for services
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Orpheus:
    """
    Orpheus-3B acoustic model + SNAC vocoder.

    ✅ Uses FAST tokenizer from the acoustic directory (tokenizer.json).
    ✅ Provides non-streaming tts() and low-latency stream_tts(...) that yields PCM chunks.

    stream_tts emits float32 mono PCM chunks via a callback while generation runs,
    so the server can forward them over WS incrementally.
    """

    BASE = 128_266
    LANE_OFFSETS = [0, 4096, 8192, 12_288, 16_384, 20_480, 24_576]

    def __init__(self, bundle_root: str, sr_out: int = 24_000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.sr = int(sr_out)

        acoustic = self._resolve_acoustic_dir(bundle_root)
        if not (acoustic.exists() and acoustic.is_dir()):
            raise ValueError(f"[tts] acoustic path must be a directory, got: {acoustic}")
        if not (acoustic / "config.json").exists():
            raise FileNotFoundError(f"[tts] Orpheus config.json not found in {acoustic}")
        if not (acoustic / "tokenizer.json").exists():
            raise FileNotFoundError(f"[tts] Orpheus tokenizer.json not found in {acoustic}")
        print(f"[tts] acoustic dir = {acoustic}")

        # FAST tokenizer (required for tokenizer.json)
        try:
            self.tok = AutoTokenizer.from_pretrained(
                str(acoustic),
                local_files_only=True,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"[tts] Failed to load FAST tokenizer from directory: {acoustic}\n"
                f"Hint: pass the ACOUSTIC DIRECTORY (not tokenizer.json file)."
            ) from e

        try:
            self.lm = AutoModelForCausalLM.from_pretrained(
                str(acoustic),
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
        except Exception as e:
            raise RuntimeError(f"[tts] Failed to load acoustic LM from {acoustic}") from e

        # Vocoder (resolve robustly)
        vocoder_dir = self._find_vocoder_dir(Path(bundle_root))
        if vocoder_dir is None:
            raise FileNotFoundError(
                f"[tts] vocoder assets not found under bundle root: {bundle_root}\n"
                f"Expected a folder like .../vocoder/ with config.json + pytorch_model.bin (or *.safetensors)."
            )
        cfg = vocoder_dir / "config.json"
        st_bin = vocoder_dir / "pytorch_model.bin"
        if not cfg.exists():
            raise FileNotFoundError(f"[tts] Missing vocoder config.json at {cfg}")
        if (not st_bin.exists()) and not list(vocoder_dir.glob("*.safetensors")):
            raise FileNotFoundError(f"[tts] Missing vocoder weights at {vocoder_dir}")
        print(f"[tts] vocoder dir = {vocoder_dir}")

        with open(cfg, "r", encoding="utf-8") as f:
            vcfg = json.load(f)

        # Try to derive frames-per-second from config; fall back to ~75 fps if unknown.
        self.sr = int(vcfg.get("sampling_rate", self.sr))
        hop = vcfg.get("hop_length")
        self.frames_per_sec = (self.sr / hop) if hop else 75.0

        from snac import SNAC
        self.voc = SNAC(**vcfg).to(self.device).eval()

        # Load weights (bin or safetensors; SNAC uses torch.load state_dict matching)
        state = None
        if st_bin.exists():
            state = torch.load(st_bin, map_location=self.device)
        else:
            st_any = sorted(vocoder_dir.glob("*.safetensors"))
            if not st_any:
                raise FileNotFoundError(f"[tts] No vocoder weights found in {vocoder_dir}")
            state = torch.load(st_any[0], map_location=self.device)

        sd = self.voc.state_dict()
        ok = {k: v for k, v in state.items() if k in sd and v.shape == sd[k].shape}
        missing = [k for k in sd.keys() if k not in ok]
        if missing:
            print(f"[tts] WARN: vocoder missing {len(missing)} keys (loading partial state)")
        self.voc.load_state_dict(ok, strict=False)

        print(f"[tts] Orpheus ready (sr={self.sr} Hz, device={self.device})")

    # ---------- path resolution ----------
    @staticmethod
    def _ensure_dir_path(p: Path) -> Path:
        try:
            if p.is_file():
                return p.parent
        except Exception:
            pass
        return p

    def _resolve_acoustic_dir(self, bundle_root: str) -> Path:
        # Prefer stable symlinks created by bootstrap
        for link in (Path("/data/models/orpheus_3b_resolved"), Path("/data/models/orpheus_resolved")):
            if link.exists() and link.is_dir():
                return link

        base = self._ensure_dir_path(Path(bundle_root))

        # Direct hit (requires tokenizer.json + weights)
        if (base / "config.json").exists() and (base / "tokenizer.json").exists() and (
            (base / "model.safetensors").exists() or any(base.glob("model-*.safetensors"))
        ):
            return base

        # Search for the shallowest valid acoustic directory
        best: Optional[Path] = None
        for cfg in base.rglob("config.json"):
            d = cfg.parent
            if (d / "tokenizer.json").exists():
                has_weights = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
                if has_weights and (best is None or len(str(d)) < len(str(best))):
                    best = d
        return best or base

    @staticmethod
    def _find_vocoder_dir(root: Path) -> Optional[Path]:
        root = Orpheus._ensure_dir_path(root)
        # Look for explicit "vocoder" first
        for d in root.rglob("vocoder"):
            if (d / "config.json").exists() and (
                (d / "pytorch_model.bin").exists() or any(d.glob("*.safetensors"))
            ):
                return d
        # Fallback: any folder that looks like a vocoder package
        for cfg in root.rglob("config.json"):
            d = cfg.parent
            if "vocoder" in str(d).lower() and (
                (d / "pytorch_model.bin").exists() or any(d.glob("*.safetensors"))
            ):
                return d
        return None

    # ---------- token → code helpers ----------
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

    # ---------- classic non-streaming ----------
    def tts(self, text_lg: str) -> np.ndarray:
        bos = self.tok.bos_token or "<s>"
        eos = self.tok.eos_token or "</s>"
        prompt = f"{bos}<|voice|>basaa_speaker<|text|>{text_lg}{eos}<|audio|>"

        enc = self.tok(prompt, return_tensors="pt", return_attention_mask=True)
        in_ids = enc["input_ids"].to(self.lm.device)
        attn = enc["attention_mask"].to(self.lm.device)

        pad_id = self.tok.eos_token_id or int(self.tok.convert_tokens_to_ids(eos))
        if getattr(self.lm.generation_config, "pad_token_id", None) is None:
            self.lm.generation_config.pad_token_id = pad_id
        if getattr(self.lm.config, "pad_token_id", None) is None:
            self.lm.config.pad_token_id = pad_id

        with torch.inference_mode():
            out = self.lm.generate(
                in_ids,
                attention_mask=attn,
                max_new_tokens=12_000,
                do_sample=False,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
            )

        tail = out.sequences[0][in_ids.shape[-1]:].tolist()
        codes, frames = self._tokens_to_codes(tail, frames_cap=4_000)
        if codes is None or frames == 0:
            return np.zeros(0, dtype=np.float32)

        with torch.inference_mode():
            wav = self.voc.decode(codes).detach().float().cpu().numpy().squeeze()

        return np.asarray(wav, dtype=np.float32).reshape(-1)

    # ---------- streaming decode ----------
    class _AudioTokenStreamer(BaseStreamer):
        def __init__(
            self,
            parent: "Orpheus",
            on_chunk: Callable[[np.ndarray], None],
            chunk_frames: int,
        ):
            super().__init__()
            self.p = parent
            self.on_chunk = on_chunk
            self.chunk_frames = max(7, int(chunk_frames))
            self.buf: List[int] = []
            self.gen_index = 0
            self.closed = False

        def put(self, value):
            """
            HF passes either:
              • LongTensor of shape (B,) or (1, N)  -> iterate over all ids
              • Python int (scalar)                 -> single id
            Normalize to a list of ints.
            """
            ids_list: List[int]
            if isinstance(value, torch.Tensor):
                ids_list = value.view(-1).tolist()
            elif isinstance(value, (list, tuple)):
                ids_list = list(value)
            else:
                # scalar
                ids_list = [int(value)]

            for tid in ids_list:
                # Only collect audio tokens; stop at first non-audio token.
                if self.p._is_audio_token(tid, self.gen_index):
                    self.buf.append(tid)
                    # When we have full frames for a chunk, decode+emit
                    frames_ready = (len(self.buf) // 7)
                    while frames_ready >= self.chunk_frames:
                        take = self.chunk_frames * 7
                        chunk = self.buf[:take]
                        del self.buf[:take]
                        self._emit_chunk(chunk)
                        frames_ready = (len(self.buf) // 7)
                    self.gen_index += 1
                else:
                    # Flush whatever full frames remain; end of audio stream.
                    frames_ready = (len(self.buf) // 7)
                    if frames_ready > 0:
                        take = frames_ready * 7
                        chunk = self.buf[:take]
                        del self.buf[:take]
                        self._emit_chunk(chunk)
                    self.closed = True
                    break

        def end(self):
            frames_ready = (len(self.buf) // 7)
            if frames_ready > 0:
                take = frames_ready * 7
                chunk = self.buf[:take]
                del self.buf[:take]
                self._emit_chunk(chunk)

        def _emit_chunk(self, tokens_7k: List[int]):
            codes, frames = self.p._tokens_to_codes(tokens_7k)
            if codes is None or frames == 0:
                return
            with torch.inference_mode():
                wav = self.p.voc.decode(codes).detach().float().cpu().numpy().squeeze()
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if wav.size:
                self.on_chunk(wav)

    def stream_tts(
        self,
        text_lg: str,
        on_chunk: Callable[[np.ndarray], None],
        chunk_ms: int = 600,
        max_new_tokens: int = 12_000,
    ):
        """
        Stream PCM chunks via callback while generating.

        on_chunk: callable(np.ndarray float32 mono) -> None
        chunk_ms: approx window length for each chunk (converted to frames with fps).
        """
        frames_per_chunk = max(7, int(self.frames_per_sec * (chunk_ms / 1000.0)))

        bos = self.tok.bos_token or "<s>"
        eos = self.tok.eos_token or "</s>"
        prompt = f"{bos}<|voice|>basaa_speaker<|text|>{text_lg}{eos}<|audio|>"

        enc = self.tok(prompt, return_tensors="pt", return_attention_mask=True)
        in_ids = enc["input_ids"].to(self.lm.device)
        attn = enc["attention_mask"].to(self.lm.device)

        pad_id = self.tok.eos_token_id or int(self.tok.convert_tokens_to_ids(eos))
        if getattr(self.lm.generation_config, "pad_token_id", None) is None:
            self.lm.generation_config.pad_token_id = pad_id
        if getattr(self.lm.config, "pad_token_id", None) is None:
            self.lm.config.pad_token_id = pad_id

        streamer = Orpheus._AudioTokenStreamer(self, on_chunk, frames_per_chunk)

        with torch.inference_mode():
            self.lm.generate(
                in_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                streamer=streamer,
            )
        # streamer.end() is called by HF
