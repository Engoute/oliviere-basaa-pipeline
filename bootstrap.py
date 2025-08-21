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


def _resolve_orpheus_dir(base: Path) -> Path:
    """Prefer acoustic model (config.json + tokenizer.json), avoid 'vocoder'."""
    if (base / "config.json").exists() and (base / "tokenizer.json").exists():
        return base
    best, best_score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        if d.name.lower() == "vocoder":
            continue
        has_tok = (d / "tokenizer.json").exists()
        has_model = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
        score = (2 if has_tok else 0) + (1 if has_model else 0)
        if score > best_score:
            best, best_score = d, score
    if best is not None:
        return best
    matches = list(base.rglob("config.json"))
    return matches[0].parent if matches else base


def _resolve_whisper_hf_dir(base: Path) -> Path | None:
    """
    Resolve an HF Whisper folder: needs config.json and either tokenizer.json/tokenizer_config.json
    or a 'processor' subfolder. Works with nested layouts.
    """
    if (base / "config.json").exists() and (
        (base / "tokenizer.json").exists()
        or (base / "tokenizer_config.json").exists()
        or (base / "processor").exists()
    ):
        return base
    best, best_score = None, -1
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        has_tok = (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()
        has_proc = (d / "processor").exists()
        score = (2 if has_tok else 0) + (1 if has_proc else 0)
        if score > best_score:
            best, best_score = d, score
    return best


def _mk_symlink(name: str, target_dir: Path):
    link = MODELS_DIR / name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target_dir, target_is_directory=True)
        print(f"[bootstrap] symlink {name} -> {target_dir}")
    except Exception as e:
        print(f"[bootstrap] WARN: could not create symlink {link} -> {target_dir}: {e}")


def _env_path(name: str, default_subdir: str) -> Path:
    """
    Read an env var; if missing, fall back to /data/models/<default_subdir>.
    Also print what we resolved so logs show exactly what paths are used.
    """
    val = os.environ.get(name, str(MODELS_DIR / default_subdir))
    p = Path(val)
    print(f"[bootstrap] {name} = {p}")
    return p


def main():
    _guard_hf_transfer()

    qwen_url    = os.environ.get("BUNDLE_QWEN_URL", "")
    whisper_url = os.environ.get("BUNDLE_WHISPER_URL", "")
    m2m_url     = os.environ.get("BUNDLE_M2M_URL", "")
    orpheus_url = os.environ.get("BUNDLE_ORPHEUS_URL", "")

    # Resolve target directories (safe defaults if envs are missing)
    path_qwen    = _env_path("PATH_QWEN",    "qwen2_5_instruct_7b")
    path_whisper = _env_path("PATH_WHISPER", "whisper_hf")
    path_m2m     = _env_path("PATH_M2M",     "m2m100_1p2B")
    path_orpheus = _env_path("PATH_ORPHEUS", "orpheus_3b")

    # Unzip bundles (idempotent)
    if qwen_url:    fetch_and_unzip(qwen_url,    path_qwen)
    if whisper_url: fetch_and_unzip(whisper_url, path_whisper)
    if m2m_url:     fetch_and_unzip(m2m_url,     path_m2m)
    if orpheus_url: fetch_and_unzip(orpheus_url, path_orpheus)

    # Orpheus: resolve & link
    orp_real = _resolve_orpheus_dir(path_orpheus)
    _mk_symlink("orpheus_3b_resolved", orp_real)
    if not (orp_real / "config.json").exists():
        print(f"[bootstrap] WARN: Orpheus config.json not found under {orp_real}")
    else:
        print(f"[bootstrap] Orpheus resolved → {orp_real}")

    # Whisper HF: resolve & link
    wh_real = _resolve_whisper_hf_dir(path_whisper)
    if wh_real is None:
        print(f"[bootstrap] ERROR: could not resolve HF Whisper under {path_whisper}")
    else:
        _mk_symlink("whisper_hf_resolved", wh_real)
        print(f"[bootstrap] Whisper HF resolved → {wh_real}")

    print("[bootstrap] Done.")


if __name__ == "__main__":
    main()
