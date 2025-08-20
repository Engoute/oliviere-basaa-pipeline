# Single image, H200-ready, CUDA 12.8, PyTorch 2.8
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1-dev wget unzip && \
    rm -rf /var/lib/apt/lists/*

# App code
WORKDIR /workspace
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App sources
COPY app/ ./app/
COPY bootstrap.py README.md ./

# Model cache + persistent volume
ENV HF_HOME=/data/cache/hf \
    TRANSFORMERS_CACHE=/data/cache/hf \
    CT2_FORCE_CPU_ISA=avx2

# Expose FastAPI
ENV HOST=0.0.0.0 \
    PORT=7860

# The following envs point to your HF dataset bundles (override at deploy if you rename)
ENV BUNDLE_QWEN_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/qwen2_5_instruct_7b_bundle_20250820_135722.zip"
ENV BUNDLE_WHISPER_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/whisper_bundle_20250817_163830.zip"
ENV BUNDLE_M2M_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/m2m100_bundle_20250817_155147.zip"
ENV BUNDLE_ORPHEUS_URL="https://huggingface.co/datasets/LeMisterIA/basaa-models/resolve/main/bundles/orpheus_bundle_20250817_155147.zip"

# Where bundles are unzipped
ENV MODELS_DIR=/data/models
ENV PATH_QWEN=$MODELS_DIR/qwen2_5_instruct_7b
ENV PATH_WHISPER_CT2=$MODELS_DIR/whisper_ct2
ENV PATH_M2M=$MODELS_DIR/m2m100_1p2B
ENV PATH_ORPHEUS=$MODELS_DIR/orpheus_3b

# Minimal persona controls
ENV PERSONA_NAME="Nkum Nyambe"
ENV PERSONA_BUILDER="Le Mister I.A (Yannick Engoute)"
ENV PERSONA_DECLINE_TECH="true"

# First run: pull + unzip bundles into /data
# (Idempotent; safe to run on every container start)
CMD python bootstrap.py && python -m uvicorn app.main:app --host $HOST --port $PORT
