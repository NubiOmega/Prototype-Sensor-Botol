"""Launch the application forcing CPU execution only."""
from __future__ import annotations

import os
import sys

# Environment variables -----------------------------------------------------
ULTRALYTICS_DEVICE_ENV = "ULTRALYTICS_DEVICE"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
PYTORCH_FORCE_CPU = "PYTORCH_FORCE_CPU"

def configure_environment() -> None:
    """Prepare environment variables to disable CUDA usage."""
    os.environ[CUDA_VISIBLE_DEVICES] = ""
    os.environ[PYTORCH_FORCE_CPU] = "1"
    os.environ[ULTRALYTICS_DEVICE_ENV] = "cpu"

def main() -> int:
    configure_environment()
    from app.main import main as app_main

    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
