import os, zipfile, shutil, tempfile, urllib.request, json
from pathlib import Path
from typing import Iterable, Optional

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _have_any(p: Path) -> bool:
    try:
        return p.exists() and any(p.iterdir())
    except Exception:
        return False

def _ensure_dir_path(p: Path) -> Path:
    """
    If 'p' is a file (e.g., .../tokenizer.json), return its parent directory.
    Otherwise return 'p' unchanged.
    """
    try:
        if p.is_file():
            print(f"[bootstrap] NOTE: path points to a FILE, using its parent: {p} -> {p.parent}")
            return p.parent
    except Exception:
        pass
    return p

def _fetch_and_unzip(url: str, target: Path):
    # handle accidental symlink path
    if target.exists() and target.is_symlink():
        try:
            target.unlink()
            print(f"[bootstrap] removed symlink at {target} to create a real dir")
        except Exception as e:
            print(f"[bootstrap] WARN: could not unlink {target}: {e}")
    target = _ensure_dir_path(target)
    target.mkdir(parents=True, exist_ok=True)
    if _have_any(target):
        print(f"[bootstrap] Exists: {target}")
        return
    print(f"[bootstrap] Downloading: {url}")
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "bundle.zip"
        with urllib.request.urlopen(url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"[bootstrap] Unzipping -> {target}")
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(target)

def _has_any(p: Path, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if list(p.glob(pat)) or list(p.rglob(pat)):
            return True
    return False

def _disable_generation_configs(root: Path):
    n = 0
    for gc in list(root.glob("generation_config.json")) + list(root.rglob("generation_config.json")):
        try:
            gc.rename(gc.with_suffix(gc.suffix + ".disabled"))
            print(f"[bootstrap] disabled {gc}")
            n += 1
        except Exception as e:
            print(f"[bootstrap] WARN: could not disable {gc}: {e}")
    if n == 0:
        print(f"[bootstrap] no generation_config.json under {root}")

def _patch_config_json(root: Path):
    cfgp = root / "config.json"
    if not cfgp.exists():
        return
    try:
        data = json.loads(cfgp.read_text())
        changed = False
        if data.get("early_stopping", None) is None:
            data["early_stopping"] = True; changed = True
        gcfg = data.get("generation_config")
        if isinstance(gcfg, dict):
            if gcfg.get("early_stopping", None) is None:
                gcfg["early_stopping"] = True; changed = True
            # drop other None values to avoid validate() errors
            for k in list(gcfg.keys()):
                if k != "early_stopping" and gcfg[k] is None:
                    del gcfg[k]; changed = True
            data["generation_config"] = gcfg
        if changed:
            cfgp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"[bootstrap] patched {cfgp} (early_stopping)")
    except Exception as e:
        print(f"[bootstrap] WARN: could not patch {cfgp}: {e}")

def _resolve_whisper_hf_dir(base: Path) -> Optional[Path]:
    def ok(d: Path) -> bool:
        return (d / "config.json").exists() and (
            (d / "processor").exists()
            or (d / "tokenizer.json").exists()
            or (d / "tokenizer_config.json").exists()
        )
    base = _ensure_dir_path(base)
    if ok(base):
        return base
    best, score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        s = (2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0) \
            + (1 if (d / "processor").exists() else 0)
        if s > score:
            best, score = d, s
    return best

def _resolve_orpheus_acoustic(base: Path) -> Optional[Path]:
    """
    Choose dir that has:
      • config.json
      • tokenizer.json
      • weights: model.safetensors or model-*.safetensors
    """
    base = _ensure_dir_path(base)
    # direct hit
    if (base / "config.json").exists() and (base / "tokenizer.json").exists() and (
        (base / "model.safetensors").exists() or any(base.glob("model-*.safetensors"))
    ):
        return base

    # search
    candidates = []
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        tok = (d / "tokenizer.json").exists()
        wt  = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
        if tok and wt:
            candidates.append(d)
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))  # prefer shallower path
        return candidates[0]

    # last resort: accept base if it at least has tokenizer.json
    return base if (base / "tokenizer.json").exists() else None

def _find_vocoder_dir(base: Path) -> Optional[Path]:
    base = _ensure_dir_path(base)
    # prefer a folder literally named "vocoder" with valid assets
    for d in base.rglob("vocoder"):
        if (d / "config.json").exists() and (
            (d / "pytorch_model.bin").exists() or any(d.glob("*.safetensors"))
        ):
            return d
    # fallback: any folder with vocoder-like assets
    for d in base.rglob("config.json"):
        vd = d.parent
        if "vocoder" in str(vd).lower() and (
            (vd / "pytorch_model.bin").exists() or any(vd.glob("*.safetensors"))
        ):
            return vd
    return None

def _mk_symlink(name: str, target_dir: Path):
    link = MODELS_DIR / name
    try:
        target_dir = _ensure_dir_path(target_dir)
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"[bootstrap] WARN: not linking {name} -> {target_dir} (missing or not a directory)")
            return
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target_dir, target_is_directory=True)
        print(f"[bootstrap] symlink {name} -> {target_dir}")
    except Exception as e:
        print(f"[bootstrap] WARN: symlink {link} -> {target_dir} failed: {e}")

# ---------- main ----------
def main():
    whisper_url  = os.environ.get("BUNDLE_WHISPER_URL", "")
    path_whisper = Path(os.environ.get("PATH_WHISPER",  str(MODELS_DIR / "whisper_hf")))

    m2m_url      = os.environ.get("BUNDLE_M2M_URL", "")
    path_m2m     = Path(os.environ.get("PATH_M2M",     str(MODELS_DIR / "m2m100_1p2B")))

    orpheus_url  = os.environ.get("BUNDLE_ORPHEUS_URL", "")
    path_orpheus = Path(os.environ.get("PATH_ORPHEUS", str(MODELS_DIR / "orpheus_3b")))

    qwen_url     = os.environ.get("BUNDLE_QWEN_URL", "")
    path_qwen    = Path(os.environ.get("PATH_QWEN",    str(MODELS_DIR / "qwen2_5_instruct_7b")))

    # Normalize any accidental FILE paths to their parent DIRs
    path_whisper = _ensure_dir_path(path_whisper)
    path_m2m     = _ensure_dir_path(path_m2m)
    path_orpheus = _ensure_dir_path(path_orpheus)
    path_qwen    = _ensure_dir_path(path_qwen)

    # Whisper
    print(f"[bootstrap] PATH_WHISPER = {path_whisper}")
    if whisper_url: _fetch_and_unzip(whisper_url, path_whisper)
    wh_real = _resolve_whisper_hf_dir(path_whisper) or path_whisper
    _mk_symlink("whisper_hf_resolved", wh_real)
    print(f"[bootstrap] Whisper HF resolved -> {wh_real}")

    # M2M
    print(f"[bootstrap] PATH_M2M = {path_m2m}")
    if m2m_url: _fetch_and_unzip(m2m_url, path_m2m)

    # Orpheus
    print(f"[bootstrap] PATH_ORPHEUS = {path_orpheus}")
    if orpheus_url: _fetch_and_unzip(orpheus_url, path_orpheus)
    ac_dir = _resolve_orpheus_acoustic(path_orpheus)
    if ac_dir:
        # Keep your original symlink AND add a generic one for future-proofing
        _mk_symlink("orpheus_3b_resolved", ac_dir)
        _mk_symlink("orpheus_resolved", ac_dir)
        print(f"[bootstrap] Orpheus acoustic resolved -> {ac_dir}")
    else:
        print(f"[bootstrap] WARN: could not resolve Orpheus acoustic dir under {path_orpheus}")
    voc_dir = _find_vocoder_dir(path_orpheus)
    if voc_dir:
        print(f"[bootstrap] Orpheus vocoder found -> {voc_dir}")
    else:
        print(f"[bootstrap] WARN: Orpheus vocoder not found under {path_orpheus}")

    # Qwen
    print(f"[bootstrap] PATH_QWEN = {path_qwen}")
    if qwen_url: _fetch_and_unzip(qwen_url, path_qwen)

    # Sanitize generation config problems (files + embedded)
    for root in [path_whisper, path_m2m, path_orpheus, path_qwen, MODELS_DIR]:
        _disable_generation_configs(root)
    _patch_config_json(path_m2m)
    _patch_config_json(path_qwen)
    if ac_dir: _patch_config_json(ac_dir)

    # sanity for M2M
    m2m_has_weights = _has_any(path_m2m, ("pytorch_model.bin", "model.safetensors", "model-*.safetensors"))
    m2m_has_tok     = _has_any(path_m2m, ("tokenizer.json", "*sentencepiece*.model", "*spm.*", "sentencepiece.bpe.model"))
    if not (m2m_has_weights and m2m_has_tok):
        print("[bootstrap] WARNING: M2M bundle may be incomplete.")

    print("[bootstrap] Done.")

if __name__ == "__main__":
    main()
