FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY app/ ./app/
COPY bootstrap.py ./bootstrap.py
# COPY README.md ./README.md  # (optional)

# Cache / models env
ENV HF_HOME=/data/cache/hf \
    TRANSFORMERS_CACHE=/data/cache/hf \
    HOST=0.0.0.0 \
    PORT=7860

# Model roots & bundles
ENV MODELS_DIR=/data/models
ENV PATH_WHISPER=$MODELS_DIR/whisper_hf
ENV BUNDLE_WHISPER_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/whisper_hf_bundle_20250821_070627.zip"

# NEW: Orpheus bundle URL (set yours here)
ENV PATH_ORPHEUS=$MODELS_DIR/orpheus_bundle \
    BUNDLE_ORPHEUS_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/orpheus_bundle_20250817_155147.zip"

EXPOSE 7860

# Download/extract bundles, then start API
CMD python bootstrap.py && python -m uvicorn app.main:app --host $HOST --port $PORT
