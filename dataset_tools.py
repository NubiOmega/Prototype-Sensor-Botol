"""Utility helpers for preparing YOLO defect datasets."""
from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced to the user
    raise SystemExit("PyYAML is required to use dataset_tools.py. Please install it with `pip install pyyaml`.") from exc

LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_folder(
    src_images: Path | str,
    src_labels: Path | str,
    out_dir: Path | str = "dataset",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, int]:
    """Split a flat folder of images/labels into train/val folders.

    Parameters
    ----------
    src_images: Path | str
        Directory containing source images (.jpg/.png).
    src_labels: Path | str
        Directory containing YOLO-format label files (.txt).
    out_dir: Path | str, optional
        Destination dataset directory (default: ``dataset``).
    train_ratio: float, optional
        Portion of samples allocated to the training set.
    seed: int, optional
        Seed used for deterministic shuffling.
    """
    image_dir = Path(src_images)
    label_dir = Path(src_labels)
    output_root = Path(out_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

    image_files = [p for p in image_dir.glob("**/*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS and p.is_file()]
    if not image_files:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    pairs: List[Tuple[Path, Path]] = []
    missing_labels: List[Path] = []
    for image_path in sorted(image_files):
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)

    orphan_labels = [
        p
        for p in label_dir.glob("**/*.txt")
        if not (image_dir / f"{p.stem}.jpg").exists()
        and not (image_dir / f"{p.stem}.png").exists()
        and not (image_dir / f"{p.stem}.jpeg").exists()
        and not (image_dir / f"{p.stem}.bmp").exists()
    ]

    if missing_labels:
        LOGGER.warning("%d images are missing matching labels", len(missing_labels))
    if orphan_labels:
        LOGGER.warning("%d label files have no matching image", len(orphan_labels))
    if not pairs:
        raise RuntimeError("No image/label pairs found; check your dataset structure")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    split_index = max(1, int(len(pairs) * train_ratio))
    if split_index >= len(pairs):
        split_index = len(pairs) - 1
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    paths = [
        output_root / "images" / "train",
        output_root / "images" / "val",
        output_root / "labels" / "train",
        output_root / "labels" / "val",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    def _copy(pair_list: Sequence[Tuple[Path, Path]], split: str) -> None:
        images_dest = output_root / "images" / split
        labels_dest = output_root / "labels" / split
        for image_src, label_src in pair_list:
            shutil.copy2(image_src, images_dest / image_src.name)
            shutil.copy2(label_src, labels_dest / label_src.name)

    _copy(train_pairs, "train")
    _copy(val_pairs, "val")

    summary = {
        "train": len(train_pairs),
        "val": len(val_pairs),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
    }
    LOGGER.info(
        "Dataset split complete: %s train / %s val (missing labels: %s, orphan labels: %s)",
        summary["train"],
        summary["val"],
        summary["missing_labels"],
        summary["orphan_labels"],
    )
    return summary


def check_dataset(path_yaml: Path | str) -> bool:
    """Validate YOLO dataset structure and provide actionable warnings."""
    yaml_path = Path(path_yaml)
    if not yaml_path.exists():
        print(f"Dataset config not found: {yaml_path}")
        return False

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(f"Unable to parse YAML: {exc}")
        return False

    if not isinstance(data, dict):
        print("YAML root must be a mapping")
        return False

    dataset_root = data.get("path", yaml_path.parent)
    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.is_absolute():
        dataset_root_path = (yaml_path.parent / dataset_root_path).resolve()

    train_rel = data.get("train", "images/train")
    val_rel = data.get("val", "images/val")
    train_images = _resolve_path(dataset_root_path, train_rel)
    val_images = _resolve_path(dataset_root_path, val_rel)
    train_labels = _mirror_labels_path(train_images)
    val_labels = _mirror_labels_path(val_images)

    ok = True
    for name, path in {
        "Images (train)": train_images,
        "Images (val)": val_images,
        "Labels (train)": train_labels,
        "Labels (val)": val_labels,
    }.items():
        if not path.exists():
            print(f"Missing {name}: {path}")
            ok = False

    names = data.get("names")
    if isinstance(names, dict):
        class_names = list(names.values())
    elif isinstance(names, list):
        class_names = names
    else:
        class_names = []
        print("Class names should be a list or dict; none found.")
        ok = False
    if not class_names:
        print("No class names defined in YAML.")
        ok = False

    ok &= _compare_pairs(train_images, train_labels, "train")
    ok &= _compare_pairs(val_images, val_labels, "val")

    if ok:
        print("Dataset looks good!")
    return ok


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _mirror_labels_path(images_path: Path) -> Path:
    if "images" in images_path.parts:
        parts = list(images_path.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_path.parent.parent / "labels" / images_path.name


def _compare_pairs(images_dir: Path, labels_dir: Path, split: str) -> bool:
    if not images_dir.exists() or not labels_dir.exists():
        return False
    images = sorted(p for p in images_dir.glob("*.*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    labels = sorted(p for p in labels_dir.glob("*.txt"))
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing = image_stems - label_stems
    orphan = label_stems - image_stems
    ok = True
    if missing:
        print(f"[{split}] Missing label files for: {', '.join(sorted(missing)[:10])}")
        if len(missing) > 10:
            print(f"[{split}] ...and {len(missing) - 10} more")
        ok = False
    if orphan:
        print(f"[{split}] Orphan label files (no matching image): {', '.join(sorted(orphan)[:10])}")
        if len(orphan) > 10:
            print(f"[{split}] ...and {len(orphan) - 10} more")
        ok = False
    if ok and not images:
        print(f"[{split}] No images found in {images_dir}")
        return False
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Helpers for preparing bottle defect datasets in YOLO format",
        epilog=(
            "Examples:\n"
            "  python dataset_tools.py --split raw_images raw_labels\n"
            "  python dataset_tools.py --check dataset/data.yaml"
        ),
    )
    parser.add_argument(
        "--split",
        nargs=2,
        metavar=("IMAGES", "LABELS"),
        help="Split raw images/labels into train/val folders under --out",
    )
    parser.add_argument(
        "--out",
        default="dataset",
        help="Destination dataset directory (default: %(default)s)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data used for training (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before the split (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        metavar="DATA_YAML",
        help="Validate a dataset YAML definition",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    exit_code = 0
    if args.split:
        images_dir, labels_dir = args.split
        try:
            summary = split_folder(images_dir, labels_dir, out_dir=args.out, train_ratio=args.train_ratio, seed=args.seed)
        except Exception as exc:  # pragma: no cover - CLI feedback
            parser.error(str(exc))
        else:
            print(
                "Split complete: {train} train / {val} val (missing labels: {missing_labels}, orphan labels: {orphan_labels})".format(
                    **summary
                )
            )
    if args.check:
        ok = check_dataset(args.check)
        if not ok:
            exit_code = 1
    if not args.split and not args.check:
        parser.print_help()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


