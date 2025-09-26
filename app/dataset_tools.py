"""Dataset helpers for reviewed annotations."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


@dataclass(slots=True)
class LabelBox:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float

    def to_yolo(self, width: float, height: float) -> Tuple[int, float, float, float, float]:
        cx = ((self.x1 + self.x2) / 2.0) / max(width, 1.0)
        cy = ((self.y1 + self.y2) / 2.0) / max(height, 1.0)
        w = abs(self.x2 - self.x1) / max(width, 1.0)
        h = abs(self.y2 - self.y1) / max(height, 1.0)
        return cx, cy, w, h


@dataclass(slots=True)
class ReviewedItem:
    inspection_id: int
    image_path: Path
    boxes: List[LabelBox]


def ensure_structure(base_dir: Path) -> Dict[str, Path]:
    images = base_dir / "images"
    labels = base_dir / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    return {"images": images, "labels": labels}


def export_reviewed_items(
    items: Sequence[ReviewedItem],
    target_dir: Path,
    classes: Iterable[str],
) -> Dict[str, Path]:
    paths = ensure_structure(target_dir)
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    exported: Dict[str, Path] = {}
    for item in items:
        if not item.image_path.exists():
            continue
        dest_image = paths["images"] / f"{item.inspection_id}_{item.image_path.name}"
        shutil.copy(item.image_path, dest_image)
        label_path = paths["labels"] / f"{dest_image.stem}.txt"
        _write_labels(label_path, dest_image, item.boxes, class_map)
        exported[str(item.inspection_id)] = dest_image
    return exported


def _write_labels(label_path: Path, image_path: Path, boxes: Sequence[LabelBox], class_map: Dict[str, int]) -> None:
    with Image.open(image_path) as img:
        width, height = img.size
    lines: List[str] = []
    for box in boxes:
        if box.cls not in class_map:
            continue
        cx, cy, w, h = box.to_yolo(width, height)
        lines.append(f"{class_map[box.cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "LabelBox",
    "ReviewedItem",
    "ensure_structure",
    "export_reviewed_items",
]

