"""Launch the application with GPU acceleration enabled when available."""
from __future__ import annotations

import os
import sys

# Environment variables -----------------------------------------------------
GPU_DEVICE_ENV = "APP_GPU_DEVICE"
ULTRALYTICS_DEVICE_ENV = "ULTRALYTICS_DEVICE"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
PYTORCH_FORCE_CPU = "PYTORCH_FORCE_CPU"

def configure_environment() -> None:
    """Prepare environment variables to favour GPU execution."""
    desired_gpu = os.environ.get(GPU_DEVICE_ENV, "0").strip()
    if desired_gpu:
        os.environ[CUDA_VISIBLE_DEVICES] = desired_gpu
    else:
        os.environ.pop(CUDA_VISIBLE_DEVICES, None)
    os.environ.pop(PYTORCH_FORCE_CPU, None)
    os.environ[ULTRALYTICS_DEVICE_ENV] = f"cuda:{desired_gpu}" if desired_gpu else "cuda"

def main() -> int:
    configure_environment()
    from app.main import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
