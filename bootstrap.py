import os, zipfile, shutil, tempfile, urllib.request
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/data/models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

def main():
    qwen_url = os.environ["BUNDLE_QWEN_URL"]
    whisper_url = os.environ["BUNDLE_WHISPER_URL"]
    m2m_url = os.environ["BUNDLE_M2M_URL"]
    orpheus_url = os.environ["BUNDLE_ORPHEUS_URL"]

    fetch_and_unzip(qwen_url, Path(os.environ["PATH_QWEN"]))
    fetch_and_unzip(whisper_url, Path(os.environ["PATH_WHISPER_CT2"]))
    fetch_and_unzip(m2m_url, Path(os.environ["PATH_M2M"]))
    fetch_and_unzip(orpheus_url, Path(os.environ["PATH_ORPHEUS"]))
    print("[bootstrap] Done.")

if __name__ == "__main__":
    main()
