# /workspace/bootstrap.py
import os
import zipfile
import shutil
import tempfile
import urllib.request
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
    """
    Prefer a directory that contains BOTH config.json and tokenizer.json (acoustic model),
    avoid folders named 'vocoder'. Fallback to the first config.json found.
    """
    if (base / "config.json").exists() and (base / "tokenizer.json").exists():
        return base

    candidates = []
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        if d.name.lower() == "vocoder":
            continue
        has_tok = (d / "tokenizer.json").exists()
        has_model = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
        score = (2 if has_tok else 0) + (1 if has_model else 0)  # prefer both present
        candidates.append((score, d))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))  # best score, then shortest path
        return candidates[0][1]

    matches = list(base.rglob("config.json"))
    return matches[0].parent if matches else base


def _mk_symlink(name: str, target_dir: Path):
    link = MODELS_DIR / name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target_dir, target_is_directory=True)
    except Exception as e:
        print(f"[bootstrap] WARN: could not create symlink {link} -> {target_dir}: {e}")


def main():
    _guard_hf_transfer()

    qwen_url = os.environ["BUNDLE_QWEN_URL"]
    whisper_url = os.environ["BUNDLE_WHISPER_URL"]
    m2m_url = os.environ["BUNDLE_M2M_URL"]
    orpheus_url = os.environ["BUNDLE_ORPHEUS_URL"]

    # unzip bundles (idempotent)
    fetch_and_unzip(qwen_url, Path(os.environ["PATH_QWEN"]))
    fetch_and_unzip(whisper_url, Path(os.environ["PATH_WHISPER_CT2"]))
    fetch_and_unzip(m2m_url, Path(os.environ["PATH_M2M"]))
    fetch_and_unzip(orpheus_url, Path(os.environ["PATH_ORPHEUS"]))

    # Orpheus: resolve to the acoustic model folder and expose a stable link
    orp_base = Path(os.environ["PATH_ORPHEUS"])
    orp_real = _resolve_model_dir(orp_base)
    _mk_symlink("orpheus_3b_resolved", orp_real)

    if not (orp_real / "config.json").exists():
        print(f"[bootstrap] WARN: Orpheus config.json not found under {orp_real}")
    else:
        print(f"[bootstrap] Orpheus resolved → {orp_real}")

    print("[bootstrap] Done.")


if __name__ == "__main__":
    main()
