# FILE: app/fast.py
import os, torch

def speed_tweaks():
    # TF32 helps matmuls on Ampere+/Hopper (safe for accuracy)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Good allocator behavior for long-running pods
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
