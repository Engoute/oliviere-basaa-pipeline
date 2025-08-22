# FILE: bootstrap.py
import os, zipfile, shutil, tempfile, urllib.request
from pathlib import Path

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

def _disable_generation_configs(root: Path):
    found = 0
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

def _resolve_whisper_hf_dir(base: Path) -> Path | None:
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

def _mk_symlink(name: str, target_dir: Path):
    link = MODELS_DIR / name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target_dir, target_is_directory=True)
        print(f"[bootstrap] symlink {name} -> {target_dir}")
    except Exception as e:
        print(f"[bootstrap] WARN: symlink {link} -> {target_dir} failed: {e}")

def main():
    whisper_url = os.environ.get("BUNDLE_WHISPER_URL", "")
    path_whisper = Path(os.environ.get("PATH_WHISPER", str(MODELS_DIR / "whisper_hf")))

    print(f"[bootstrap] PATH_WHISPER = {path_whisper}")
    if whisper_url:
        _fetch_and_unzip(whisper_url, path_whisper)

    wh_real = _resolve_whisper_hf_dir(path_whisper)
    if wh_real is None:
        print(f"[bootstrap] ERROR: could not resolve HF Whisper under {path_whisper}")
    else:
        _mk_symlink("whisper_hf_resolved", wh_real)
        print(f"[bootstrap] Whisper HF resolved -> {wh_real}")

    # disable problematic generation_config.json BEFORE app import
    for root in [path_whisper, MODELS_DIR]:
        _disable_generation_configs(root)

    print("[bootstrap] Done.")

if __name__ == "__main__":
    main()
