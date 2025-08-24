# FILE: app/tts_orpheus.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Callable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer


class Orpheus:
    """
    Orpheus-3B acoustic model + SNAC vocoder.

    ✅ Uses FAST tokenizer from the acoustic directory (tokenizer.json).
    ✅ Provides non-streaming tts() and NEW low-latency stream_tts(...) that yields PCM chunks.

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
        assert (acoustic / "config.json").exists(), f"[tts] Orpheus config.json not found in {acoustic}"

        # FAST tokenizer (required for tokenizer.json)
        self.tok = AutoTokenizer.from_pretrained(
            str(acoustic),
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )

        self.lm = AutoModelForCausalLM.from_pretrained(
            str(acoustic),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        # Vocoder
        vocoder_dir = Path(bundle_root) / "vocoder"
        cfg = vocoder_dir / "config.json"
        st  = vocoder_dir / "pytorch_model.bin"
        assert cfg.exists() and st.exists(), f"[tts] vocoder assets missing under {vocoder_dir}"

        with open(cfg, "r", encoding="utf-8") as f:
            vcfg = json.load(f)
        # Try to derive frames-per-second from config; fall back to ~75 fps if unknown.
        self.sr = int(vcfg.get("sampling_rate", self.sr))
        hop = vcfg.get("hop_length")
        self.frames_per_sec = (self.sr / hop) if hop else 75.0

        from snac import SNAC
        self.voc = SNAC(**vcfg).to(self.device).eval()
        state = torch.load(st, map_location=self.device)
        sd = self.voc.state_dict()
        ok = {k: v for k, v in state.items() if k in sd and v.shape == sd[k].shape}
        self.voc.load_state_dict(ok, strict=False)

        print(f"[tts] Orpheus ready (sr={self.sr} Hz)")

    # ---------- path resolution ----------
    def _resolve_acoustic_dir(self, bundle_root: str) -> Path:
        symlink = Path("/data/models/orpheus_3b_resolved")
        if symlink.exists() and symlink.is_dir():
            return symlink

        base = Path(bundle_root)
        if (base / "config.json").exists() and (base / "tokenizer.json").exists():
            return base

        best: Optional[Path] = None
        for cfg in base.rglob("config.json"):
            d = cfg.parent
            if (d / "tokenizer.json").exists():
                has_weights = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
                if has_weights and (best is None or len(str(d)) < len(str(best))):
                    best = d
        return best or base

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

    # ---------- NEW: streaming decode ----------
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
            # value: LongTensor [1, N] newly generated ids
            ids = value.tolist()[0]
            for tid in ids:
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
            # Final flush (in case model ended exactly on frame boundary)
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
            # Ensure 1-D float32
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
        # Approx frames per chunk
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
            # This call runs until generation ends; streamer.put() is called along the way.
            self.lm.generate(
                in_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                streamer=streamer,
            )
        # streamer.end() is invoked by HF generate()
