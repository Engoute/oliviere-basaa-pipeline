import os, zipfile, shutil, tempfile, urllib.request
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _guard_hf_transfer():
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1":
        try:
            import hf_transfer  # noqa: F401
        except Exception:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            print("[bootstrap] hf_transfer missing; disabling acceleration.")

def have_any(p: Path) -> bool:
    return p.exists() and any(p.rglob("*"))

def fetch_and_unzip(url: str, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    if have_any(target):
        print(f"[bootstrap] Exists: {target}")
        return
    print(f"[bootstrap] Downloading: {url}")
    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "bundle.zip"
        with urllib.request.urlopen(url) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"[bootstrap] Unzipping to {target}")
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(target)

def _resolve_model_dir(base: Path) -> Path:
    """Return a subdir (or self) that actually contains config.json."""
    if (base / "config.json").exists():
        return base
    matches = list(base.rglob("config.json"))
    return matches[0].parent if matches else base

def main():
    _guard_hf_transfer()

    qwen_url    = os.environ["BUNDLE_QWEN_URL"]
    whisper_url = os.environ["BUNDLE_WHISPER_URL"]
    m2m_url     = os.environ["BUNDLE_M2M_URL"]
    orpheus_url = os.environ["BUNDLE_ORPHEUS_URL"]

    # unzip bundles (idempotent)
    fetch_and_unzip(qwen_url,    Path(os.environ["PATH_QWEN"]))
    fetch_and_unzip(whisper_url, Path(os.environ["PATH_WHISPER_CT2"]))
    fetch_and_unzip(m2m_url,     Path(os.environ["PATH_M2M"]))
    fetch_and_unzip(orpheus_url, Path(os.environ["PATH_ORPHEUS"]))

    # Orpheus: create a stable, resolved path -> /data/models/orpheus_3b_resolved
    orp_base = Path(os.environ["PATH_ORPHEUS"])
    orp_real = _resolve_model_dir(orp_base)
    link = MODELS_DIR / "orpehus_3b_resolved_tmp"  # temp to avoid partial updates
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(orp_real, target_is_directory=True)

    final = MODELS_DIR / "orpehus_3b_resolved"
    if final.exists() or final.is_symlink():
        final.unlink()
    link.rename(final)

    if not (final / "config.json").exists():
        print(f"[bootstrap] WARN: Orpheus config.json not found under {final}")
    else:
        print(f"[bootstrap] Orpheus resolved → {orp_real}")

    print("[bootstrap] Done.")

if __name__ == "__main__":
    main()
