FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# ---- Base env ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- System deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip ca-certificates libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- Python deps ----
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y torchvision || true

# ---- App code ----
COPY app/ ./app/
COPY bootstrap.py ./bootstrap.py

# ---- Caches / server ----
ENV HF_HOME=/data/cache/hf \
    TRANSFORMERS_CACHE=/data/cache/hf \
    HOST=0.0.0.0 \
    PORT=7860

# ===== Models (paths + bundles) =====
ENV MODELS_DIR=/data/models

# Basaa-finetuned Whisper (existing)
ENV PATH_WHISPER_BASAA=$MODELS_DIR/whisper_hf
ENV BUNDLE_WHISPER_BASAA_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/whisper_hf_bundle_20250821_070627.zip"

# Whisper v3 (general) — your uploaded ZIP
ENV PATH_WHISPER_GENERAL=$MODELS_DIR/whisper_general
ENV BUNDLE_WHISPER_GENERAL_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/asr/whisper_v3_general_20250825_223803.zip"

# Orpheus bundle
ENV PATH_ORPHEUS=$MODELS_DIR/orpheus_bundle
ENV BUNDLE_ORPHEUS_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/orpheus_bundle_20250817_155147.zip"

# M2M-100 + Qwen (paths usually pre-mounted; bundle URLs optional)
ENV PATH_M2M=$MODELS_DIR/m2m100_1p2B
ENV BUNDLE_M2M_URL=""
ENV PATH_QWEN=$MODELS_DIR/qwen2_5_instruct_7b
ENV BUNDLE_QWEN_URL=""

# Legacy back-compat
ENV PATH_WHISPER=$PATH_WHISPER_BASAA

EXPOSE 7860

# ---- Launch ----
CMD bash -lc "python bootstrap.py && python -m uvicorn app.main:app --host ${HOST} --port ${PORT}"
