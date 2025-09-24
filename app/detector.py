"""Ultralytics YOLO detector wrapper."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "yolo11n.pt"
DEVICE_ENV_VAR = "ULTRALYTICS_DEVICE"


class Detector:
    """Encapsulate YOLO model loading and inference."""

    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._model_path: str = ""
        self._names: Dict[int, str] = {}
        self._device: Optional[str] = self._resolve_device()

    @property
    def names(self) -> Dict[int, str]:
        return self._names

    @property
    def model_path(self) -> str:
        return self._model_path

    def _resolve_device(self) -> Optional[str]:
        value = os.environ.get(DEVICE_ENV_VAR, "").strip()
        return value or None

    def _apply_device(self) -> None:
        if not self._device or self._model is None:
            return
        try:
            self._model.to(self._device)
        except Exception as exc:
            LOGGER.warning("Unable to move model to device '%s': %s", self._device, exc)
            raise

    def load(self, model_path: Optional[str] = None) -> List[str]:
        """Load model from supplied path or default."""
        target = model_path or DEFAULT_MODEL
        self._device = self._resolve_device()
        resolved = Path(target)
        try:
            self._model = YOLO(str(resolved))
        except Exception:  # pragma: no cover - bubbled to UI
            if resolved.exists():
                raise
            LOGGER.info("Falling back to default YOLO model: %s", DEFAULT_MODEL)
            self._model = YOLO(DEFAULT_MODEL)
            target = DEFAULT_MODEL
        self._model_path = target
        self._apply_device()
        self._names = self._model.model.names if hasattr(self._model, "model") else self._model.names
        return list(self._names.values())

    def predict(
        self,
        frame: np.ndarray,
        confidence: float,
        class_filter: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, List[Dict[str, float]], float]:
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        start = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=float(confidence),
            verbose=False,
            classes=list(class_filter) if class_filter else None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = results[0]
        annotated = frame.copy()
        detections: List[Dict[str, float]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return annotated, detections, elapsed_ms
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.astype(int)
            conf = float(confidences[idx])
            cls_id = int(classes[idx])
            label = self._names.get(cls_id, str(cls_id))
            detections.append({
                "label": label,
                "confidence": conf,
            })
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(annotated, text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return annotated, detections, elapsed_ms

    def class_ids_from_labels(self, labels: Iterable[str]) -> List[int]:
        lookup = {name: idx for idx, name in self._names.items()}
        return [lookup[name] for name in labels if name in lookup]
