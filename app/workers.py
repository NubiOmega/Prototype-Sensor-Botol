"""Worker threads for capture and detection."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from .camera import CameraManager
from .detector import Detector
from .utils import FpsCalculator, RECORD_DIR, SNAPSHOT_DIR, ensure_dirs, timestamp_for_file

LOGGER = logging.getLogger(__name__)


class VideoWorker(QObject):
    frameReady = Signal(np.ndarray)
    statsUpdated = Signal(float, float)
    logMessage = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        camera: CameraManager,
        detector: Optional[Detector],
        max_iterations: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.camera = camera
        self.detector = detector
        self._running = False
        self._detection_enabled = False
        self._confidence = 0.25
        self._class_ids: List[int] = []
        self._record_enabled = False
        self._writer: cv2.VideoWriter | None = None
        self._writer_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._fps = FpsCalculator()
        self._pending_snapshot = False
        self._max_iterations = max_iterations
        self._iterations = 0

    def start(self) -> None:
        with self._lock:
            self._running = True
        ensure_dirs()
        self.logMessage.emit("Video worker started")

    def request_stop(self) -> None:
        with self._lock:
            self._running = False
            self._detection_enabled = False
            self._record_enabled = False
            self._pending_snapshot = False

    def enable_detection(self, confidence: float, class_ids: Iterable[int], record: bool) -> None:
        with self._lock:
            self._confidence = confidence
            self._class_ids = list(class_ids)
            self._detection_enabled = True
            self._record_enabled = record
        self.logMessage.emit(
            f"Detection enabled (conf={confidence:.2f}, classes={self._class_ids or 'all'}, record={record})"
        )

    def disable_detection(self) -> None:
        with self._lock:
            self._detection_enabled = False
            self._record_enabled = False
        self._tear_down_writer()
        self.logMessage.emit("Detection disabled")

    def request_snapshot(self) -> None:
        with self._lock:
            self._pending_snapshot = True

    def update_detector(self, detector: Detector) -> None:
        with self._lock:
            self.detector = detector

    def run(self) -> None:
        self.start()
        inference_ms = 0.0
        while True:
            with self._lock:
                running = self._running
                detection_enabled = self._detection_enabled
                confidence = self._confidence
                class_ids = list(self._class_ids)
                record_enabled = self._record_enabled
                pending_snapshot = self._pending_snapshot
                detector = self.detector
            if not running:
                break

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            annotated = frame
            if detection_enabled and detector:
                try:
                    annotated, _, inference_ms = detector.predict(frame, confidence, class_ids or None)
                except Exception as exc:  # pragma: no cover - runtime safety
                    LOGGER.exception("Detector failed")
                    self.error.emit(f"Detection error: {exc}")
                    self.disable_detection()
                    detection_enabled = False
                    inference_ms = 0.0
            else:
                inference_ms = 0.0

            fps = self._fps.update()
            self.frameReady.emit(annotated)
            self.statsUpdated.emit(fps, inference_ms)

            if record_enabled:
                self._write_frame(annotated)
            if pending_snapshot:
                self._save_snapshot(annotated)
                with self._lock:
                    self._pending_snapshot = False

            self._iterations += 1
            if self._max_iterations and self._iterations >= self._max_iterations:
                break
        self._finalize()

    def _save_snapshot(self, frame: np.ndarray) -> None:
        filename = SNAPSHOT_DIR / f"snapshot_{timestamp_for_file()}.jpg"
        cv2.imwrite(str(filename), frame)
        self.logMessage.emit(f"Snapshot saved to {filename}")

    def _write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._create_writer(frame)
        if self._writer:
            self._writer.write(frame)

    def _create_writer(self, frame: np.ndarray) -> None:
        ensure_dirs()
        height, width = frame.shape[:2]
        filename = RECORD_DIR / f"record_{timestamp_for_file()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filename), fourcc, max(1.0, self._fps.current_fps or 15.0), (width, height))
        if not writer.isOpened():
            self.logMessage.emit("Falling back to AVI writer")
            filename = RECORD_DIR / f"record_{timestamp_for_file()}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(filename), fourcc, max(1.0, self._fps.current_fps or 15.0), (width, height))
        if writer.isOpened():
            self._writer = writer
            self._writer_path = filename
            self.logMessage.emit(f"Recording to {filename}")
        else:
            self.error.emit("Unable to start recording writer")

    def _tear_down_writer(self) -> None:
        if self._writer:
            self._writer.release()
            self.logMessage.emit(f"Recording stopped ({self._writer_path})")
        self._writer = None
        self._writer_path = None

    def _finalize(self) -> None:
        self._tear_down_writer()
        self.camera.release()
        self.logMessage.emit("Video worker finished")
        self.finished.emit()


class ModelLoaderWorker(QObject):
    """Background loader for YOLO models with user-friendly messaging."""

    modelLoaded = Signal(object, list, str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    def run(self) -> None:
        detector = Detector()
        try:
            classes = detector.load(self.model_path)
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("Model loading failed")
            self.error.emit(str(exc))
        else:
            self.modelLoaded.emit(detector, classes, detector.model_path)
        finally:
            self.finished.emit()
