import gc
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
# fp16 only on CUDA (autocast keeps fp32 weights, so ops that dislike fp16 fall back safely).
HALF = DEVICE.type == "cuda"
if HALF:
    torch.backends.cuda.matmul.allow_tf32 = True


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=HALF)


def clear_cache():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        torch.mps.empty_cache()


def device_label() -> str:
    if DEVICE.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)}, fp16)"
    return f"{DEVICE.type.upper()} (fp32)"
