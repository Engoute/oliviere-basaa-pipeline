# FILE: bootstrap.py
import os, zipfile, shutil, tempfile, urllib.request
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _have_any(p: Path) -> bool:
    try:
        return p.exists() and any(p.iterdir())
    except Exception:
        return False

def _fetch_and_unzip(url: str, target: Path):
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

def _env_path(name: str, default_subdir: str) -> Path:
    p = Path(os.environ.get(name, str(MODELS_DIR / default_subdir)))
    print(f"[bootstrap] {name} = {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _mk_symlink(name: str, target_dir: Path):
    link = MODELS_DIR / name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target_dir, target_is_directory=True)
        print(f"[bootstrap] symlink {name} -> {target_dir}")
    except Exception as e:
        print(f"[bootstrap] WARN: symlink {link} -> {target_dir} failed: {e}")

def _resolve_orpheus_dir(base: Path) -> Path:
    # Pick acoustic model dir (has config.json + tokenizer.json); skip 'vocoder'
    if (base / "config.json").exists() and (base / "tokenizer.json").exists():
        return base
    best, score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        if d.name.lower() == "vocoder":
            continue
        tok = (d / "tokenizer.json").exists()
        model = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
        s = (2 if tok else 0) + (1 if model else 0)
        if s > score:
            best, score = d, s
    return best or base

def _resolve_whisper_hf_dir(base: Path) -> Optional[Path]:
    # Look for HF-style bundle (config + tokenizer or processor/)
    def is_hf(d: Path) -> bool:
        return (d / "config.json").exists() and (
            (d / "processor").exists()
            or (d / "tokenizer.json").exists()
            or (d / "tokenizer_config.json").exists()
        )
    if is_hf(base):
        return base
    best, score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        s = (2 if (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists() else 0) \
            + (1 if (d / "processor").exists() else 0)
        if s > score:
            best, score = d, s
    return best

def _disable_generation_configs(root: Path):
    found = 0
    # explicit shallow + recursive to be safe across odd FS behaviors
    paths = list(root.glob("generation_config.json")) + list(root.rglob("generation_config.json"))
    for gc in paths:
        try:
            newp = gc.with_suffix(gc.suffix + ".disabled")
            gc.rename(newp)
            print(f"[bootstrap] disabled {gc}")
            found += 1
        except Exception as e:
            print(f"[bootstrap] WARN: could not disable {gc}: {e}")
    if found == 0:
        print(f"[bootstrap] no generation_config.json under {root}")

def main():
    # Resolve env
    qwen_url    = os.environ.get("BUNDLE_QWEN_URL", "")
    whisper_url = os.environ.get("BUNDLE_WHISPER_URL", "")
    m2m_url     = os.environ.get("BUNDLE_M2M_URL", "")
    orpheus_url = os.environ.get("BUNDLE_ORPHEUS_URL", "")

    path_qwen    = _env_path("PATH_QWEN",    "qwen2_5_instruct_7b")
    path_whisper = _env_path("PATH_WHISPER", "whisper_hf")
    path_m2m     = _env_path("PATH_M2M",     "m2m100_1p2B")
    path_orpheus = _env_path("PATH_ORPHEUS", "orpheus_3b")

    # Download + unzip if empty
    if qwen_url:    _fetch_and_unzip(qwen_url,    path_qwen)
    if whisper_url: _fetch_and_unzip(whisper_url, path_whisper)
    if m2m_url:     _fetch_and_unzip(m2m_url,     path_m2m)
    if orpheus_url: _fetch_and_unzip(orpheus_url, path_orpheus)

    # Symlinks to canonical resolved dirs
    orp_real = _resolve_orpheus_dir(path_orpheus)
    _mk_symlink("orpheus_3b_resolved", orp_real)
    print(f"[bootstrap] Orpheus resolved -> {orp_real}")

    wh_real = _resolve_whisper_hf_dir(path_whisper)
    if wh_real is None:
        print(f"[bootstrap] ERROR: could not resolve HF Whisper under {path_whisper}")
    else:
        _mk_symlink("whisper_hf_resolved", wh_real)
        print(f"[bootstrap] Whisper HF resolved -> {wh_real}")

    # HARD FIX: disable generation_config.json everywhere BEFORE app import
    for root in [path_qwen, path_whisper, path_m2m, orp_real, MODELS_DIR]:
        _disable_generation_configs(root)

    print("[bootstrap] Done.")

if __name__ == "__main__":
    main()
