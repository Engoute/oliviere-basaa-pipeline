# Basaa Realtime Pipeline (single image)

Models:
- Whisper large-v3 (CT2, your finetune)
- M2M-100 1.2B (your finetune)
- Qwen2.5-Instruct 7B (local)
- Orpheus 3B + SNAC (24k)

## Build locally
```bash
docker build -t basaa-pipeline:latest .
docker run --gpus all --rm -p 7860:7860 \
  -v /your/host/volume:/data \
  -e BUNDLE_QWEN_URL=... \
  -e BUNDLE_WHISPER_URL=... \
  -e BUNDLE_M2M_URL=... \
  -e BUNDLE_ORPHEUS_URL=... \
  basaa-pipeline:latest
