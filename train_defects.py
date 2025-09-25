"""Command line interface for training YOLO bottle defect models."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency at runtime
    YOLO = None

LOGGER = logging.getLogger("train")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Train an Ultralytics YOLO model for glass bottle defect detection",
        epilog="Example: python train_defects.py --data dataset/data.yaml --model yolo11n.pt --device 0",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    parser.add_argument("--model", default="yolo11n.pt", help="Base model weights to start from")
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset YAML definition")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. 'cpu' or '0' for first GPU")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--project", default="runs/detect", help="Project directory for Ultralytics outputs")
    parser.add_argument("--name", default="train", help="Run name inside the project directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--noval", action="store_true", help="Disable validation to speed up training")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging output")
    return parser.parse_args(list(argv) if argv is not None else None)


def _should_enable_amp(device: str) -> bool:
    normalized = device.strip().lower()
    if normalized in {"0", "cuda", "cuda:0"} and torch is not None:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
    return False


def train(args: argparse.Namespace) -> Path:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if YOLO is None:
        LOGGER.error("Ultralytics is not installed. Install with `pip install ultralytics`.")
        raise SystemExit(2)

    if not Path(args.data).exists():
        LOGGER.error("Dataset YAML not found: %s", args.data)
        raise SystemExit(1)

    try:
        model = YOLO(args.model)
    except Exception as exc:  # pragma: no cover - surfaced to CLI user
        LOGGER.error("Failed to load model %s: %s", args.model, exc)
        raise SystemExit(2) from exc

    train_kwargs: Dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "save_period": 1,
    }
    if args.noval:
        train_kwargs["val"] = False
    if _should_enable_amp(args.device):
        train_kwargs["amp"] = True
        LOGGER.info("CUDA detected - automatic AMP enabled")

    LOGGER.info(
        "Starting training with parameters: %s",
        {k: v for k, v in train_kwargs.items() if k not in {"project", "name"}} | {"run_dir": Path(args.project) / args.name},
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message and "cuda" in message:
            LOGGER.error("CUDA out of memory. Try reducing --imgsz or --batch size.")
            raise SystemExit(3) from exc
        raise

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = Path(args.project) / args.name
    else:
        save_dir = Path(save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        print(f"Training complete. Best weights saved to: {best_weights}")
    else:
        print(f"Training finished, but best weights not found at expected path: {best_weights}")
    return best_weights


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        train(args)
    except SystemExit as exc:
        return int(exc.code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





