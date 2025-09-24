"""General utilities for the object detection app."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PySide6.QtGui import QImage


RUNS_DIR = Path("runs")
SNAPSHOT_DIR = RUNS_DIR / "snapshots"
RECORD_DIR = RUNS_DIR / "records"


def ensure_dirs() -> None:
    """Create runtime output directories."""
    for directory in (RUNS_DIR, SNAPSHOT_DIR, RECORD_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def timestamp_for_file() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class FpsCalculator:
    """Simple moving-average FPS calculator."""

    def __init__(self, window_size: int = 30):
        self.window_size = max(1, window_size)
        self.samples: List[float] = []
        self._last_time = None

    def update(self) -> float:
        """Update with current time and return smoothed FPS."""
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return 0.0
        delta = now - self._last_time
        self._last_time = now
        if delta <= 0:
            return self.current_fps
        fps = 1.0 / delta
        self.samples.append(fps)
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
        return self.current_fps

    @property
    def current_fps(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


def numpy_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a BGR numpy frame into a QImage for display."""
    if frame.ndim != 3:
        raise ValueError("Expected color frame with 3 dimensions")
    rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    return QImage(
        rgb_frame.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    ).copy()


def format_iterable(values: Iterable[str]) -> str:
    return ", ".join(values)
