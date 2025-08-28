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
    try:
        if p.is_file():
            print(f"[bootstrap] NOTE: path points to a FILE, using its parent: {p} -> {p.parent}")
            return p.parent
    except Exception:
        pass
    return p

def _download_url(url: str, dst: Path):
    token = os.environ.get("HF_TOKEN", "").strip()
    req = urllib.request.Request(url)
    if token and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)

def _fetch_and_unzip(url: str, target: Path):
    if not url:
        return
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
        _download_url(url, zpath)
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
    try:
        paths = {p.resolve() for p in root.rglob("generation_config.json")}
        for gc in sorted(paths):
            try:
                dst = gc.with_suffix(gc.suffix + ".disabled")
                if dst.exists():
                    continue
                gc.rename(dst)
                print(f"[bootstrap] disabled {gc}")
                n += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"[bootstrap] WARN: could not disable {gc}: {e}")
    except Exception as e:
        print(f"[bootstrap] WARN: generation config scan failed under {root}: {e}")
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
    base = _ensure_dir_path(base)
    if (base / "config.json").exists() and (base / "tokenizer.json").exists() and (
        (base / "model.safetensors").exists() or any(base.glob("model-*.safetensors"))
    ):
        return base
    candidates = []
    for cfg in base.rglob("config.json"):
        d = cfg.parent
        tok = (d / "tokenizer.json").exists()
        wt  = (d / "model.safetensors").exists() or any(d.glob("model-*.safetensors"))
        if tok and wt:
            candidates.append(d)
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        return candidates[0]
    return base if (base / "tokenizer.json").exists() else None

def _find_vocoder_dir(base: Path) -> Optional[Path]:
    base = _ensure_dir_path(base)
    for d in base.rglob("vocoder"):
        if (d / "config.json").exists() and (
            (d / "pytorch_model.bin").exists() or any(d.glob("*.safetensors"))
        ):
            return d
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
    basaa_url  = os.environ.get("BUNDLE_WHISPER_BASAA_URL", "")
    path_basaa = Path(os.environ.get("PATH_WHISPER_BASAA", str(MODELS_DIR / "whisper_hf")))

    general_url  = os.environ.get("BUNDLE_WHISPER_GENERAL_URL", "")
    path_general = Path(os.environ.get("PATH_WHISPER_GENERAL", str(MODELS_DIR / "whisper_general")))

    llava_url  = os.environ.get("BUNDLE_LLAVA_VIDEO_URL", "")
    path_llava = Path(os.environ.get("PATH_LLAVA_VIDEO", str(MODELS_DIR / "llava_next_video")))

    m2m_url      = os.environ.get("BUNDLE_M2M_URL", "")
    path_m2m     = Path(os.environ.get("PATH_M2M",     str(MODELS_DIR / "m2m100_1p2B")))

    orpheus_url  = os.environ.get("BUNDLE_ORPHEUS_URL", "")
    path_orpheus = Path(os.environ.get("PATH_ORPHEUS", str(MODELS_DIR / "orpheus_bundle")))

    qwen_url     = os.environ.get("BUNDLE_QWEN_URL", "")
    path_qwen    = Path(os.environ.get("PATH_QWEN",    str(MODELS_DIR / "qwen2_5_instruct_7b")))

    path_basaa   = _ensure_dir_path(path_basaa)
    path_general = _ensure_dir_path(path_general)
    path_llava   = _ensure_dir_path(path_llava)
    path_m2m     = _ensure_dir_path(path_m2m)
    path_orpheus = _ensure_dir_path(path_orpheus)
    path_qwen    = _ensure_dir_path(path_qwen)

    print(f"[bootstrap] PATH_WHISPER_BASAA = {path_basaa}")
    if basaa_url: _fetch_and_unzip(basaa_url, path_basaa)
    basaa_real = _resolve_whisper_hf_dir(path_basaa) or path_basaa
    _mk_symlink("whisper_hf_resolved", basaa_real)
    _mk_symlink("whisper_basaa_resolved", basaa_real)
    print(f"[bootstrap] Whisper Basaa resolved -> {basaa_real}")

    print(f"[bootstrap] PATH_WHISPER_GENERAL = {path_general}")
    if general_url: _fetch_and_unzip(general_url, path_general)
    general_real = _resolve_whisper_hf_dir(path_general) or path_general
    _mk_symlink("whisper_general_resolved", general_real)
    print(f"[bootstrap] Whisper General resolved -> {general_real}")

    print(f"[bootstrap] PATH_LLAVA_VIDEO = {path_llava}")
    if llava_url: _fetch_and_unzip(llava_url, path_llava)
    _mk_symlink("llava_next_video_resolved", path_llava)

    print(f"[bootstrap] PATH_M2M = {path_m2m}")
    if m2m_url: _fetch_and_unzip(m2m_url, path_m2m)

    print(f"[bootstrap] PATH_ORPHEUS = {path_orpheus}")
    if orpheus_url: _fetch_and_unzip(orpheus_url, path_orpheus)
    ac_dir = _resolve_orpheus_acoustic(path_orpheus)
    if ac_dir:
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

    print(f"[bootstrap] PATH_QWEN = {path_qwen}")
    if qwen_url: _fetch_and_unzip(qwen_url, path_qwen)

    for root in [path_basaa, path_general, path_llava, path_m2m, path_orpheus, path_qwen, MODELS_DIR]:
        _disable_generation_configs(root)
    _patch_config_json(path_m2m); _patch_config_json(path_qwen)
    if ac_dir: _patch_config_json(ac_dir)

    m2m_has_weights = _has_any(path_m2m, ("pytorch_model.bin", "model.safetensors", "model-*.safetensors"))
    m2m_has_tok     = _has_any(path_m2m, ("tokenizer.json", "*sentencepiece*.model", "*spm.*", "sentencepiece.bpe.model"))
    if not (m2m_has_weights and m2m_has_tok):
        print("[bootstrap] WARNING: M2M bundle may be incomplete.")

    print("[bootstrap] Done.")

if __name__ == "__main__":
    main()
