"""Worker threads for capture, detection, and training."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread

from ultralytics import YOLO

from .camera import CameraManager
from .detector import Detector
from .utils import FpsCalculator, RECORD_DIR, SNAPSHOT_DIR, ensure_dirs, timestamp_for_file

try:  # pragma: no cover - optional dependency at runtime
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

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


class TrainingWorker(QThread):
    """Background trainer that streams Ultralytics logs and metrics."""

    on_log = Signal(str)
    on_progress = Signal(int, float, float)
    on_finished = Signal(str)
    on_error = Signal(str)

    def __init__(self, params: Dict[str, Any], log_path: Path) -> None:
        super().__init__()
        self.params = params
        self.log_path = log_path
        self._stop_requested = False
        self._best_map50 = 0.0
        self._best_map5095 = 0.0
        self._stop_notified = False
        self._log_callback: Optional[Callable[[str], None]] = None

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:  # noqa: D401
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            def emit_log(message: str) -> None:
                timestamp = datetime.now().strftime("%H:%M:%S")
                line = f"[{timestamp}] {message}"
                self.on_log.emit(line)
                log_file.write(line + "\n")
                log_file.flush()

            self._log_callback = emit_log
            self._stop_notified = False
            emit_log("Training worker started")

            try:
                model = YOLO(self.params["model"])
            except Exception as exc:  # pragma: no cover - runtime safety
                LOGGER.exception("Failed to initialise YOLO model")
                self._log(f"Failed to load model: {exc}")
                self.on_error.emit(str(exc))
                self._log_callback = None
                return

            train_kwargs = {k: v for k, v in self.params.items() if k != "model"}
            resume = bool(train_kwargs.pop("resume", False))
            noval = bool(train_kwargs.pop("noval", False))
            train_kwargs.setdefault("project", "runs/detect")
            train_kwargs.setdefault("name", "train")
            train_kwargs["resume"] = resume
            train_kwargs["close_mosaic"] = 1
            if noval:
                train_kwargs["val"] = False
            if self._should_enable_amp(train_kwargs.get("device", "cpu")):
                train_kwargs["amp"] = True
                self._log("AMP enabled for CUDA device")

            self._log(f"Training parameters: {train_kwargs}")

            ul_logger = logging.getLogger("ultralytics")
            handler = _SignalLogHandler(self._log)
            handler.setLevel(logging.INFO)
            ul_logger.addHandler(handler)
            previous_level = ul_logger.level
            if previous_level > logging.INFO:
                ul_logger.setLevel(logging.INFO)

            model.add_callback("on_fit_epoch_end", self._on_epoch_end)

            try:
                results = model.train(**train_kwargs)
            except RuntimeError as exc:  # pragma: no cover - surfaced to UI
                self._log(f"Training failed: {exc}")
                if "out of memory" in str(exc).lower() and "cuda" in str(exc).lower():
                    self._log("Detected CUDA OOM. Consider reducing image size or batch size.")
                self.on_error.emit(str(exc))
                self._log_callback = None
                return
            except Exception as exc:  # pragma: no cover - surfaced to UI
                LOGGER.exception("Training crashed")
                self._log(f"Training crashed: {exc}")
                self.on_error.emit(str(exc))
                self._log_callback = None
                return
            finally:
                try:
                    model.callbacks.clear()
                except Exception:  # pragma: no cover - cleanup best-effort
                    pass
                ul_logger.removeHandler(handler)
                ul_logger.setLevel(previous_level)

            save_dir_attr = getattr(results, "save_dir", None)
            if save_dir_attr is None:
                save_dir = Path(train_kwargs.get("project", "runs/detect")) / train_kwargs.get("name", "train")
            else:
                save_dir = Path(save_dir_attr)
            if not save_dir.is_absolute():
                save_dir = Path(train_kwargs.get("project", "runs/detect")) / train_kwargs.get("name", "train")
            best_path = save_dir / "weights" / "best.pt"
            self._log(f"Training finished. Best weights expected at: {best_path}")
            self.on_finished.emit(str(best_path))
            self._log_callback = None

    def _log(self, message: str) -> None:
        if self._log_callback:
            self._log_callback(message)
        else:
            self.on_log.emit(message)

    def _should_enable_amp(self, device: str) -> bool:
        normalized = str(device).strip().lower()
        if normalized in {"0", "cuda", "cuda:0"} and torch is not None:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return True
        return False

    def _on_epoch_end(self, trainer: Any) -> None:
        epoch_index = int(getattr(trainer, "epoch", -1))
        epoch = epoch_index + 1
        metrics = getattr(trainer, "metrics", {}) or {}
        map50 = self._extract_metric(metrics, [
            "metrics/mAP50(B)",
            "metrics/mAP50(R)",
            "map50",
        ])
        map5095 = self._extract_metric(metrics, [
            "metrics/mAP50-95(B)",
            "metrics/mAP50-95(R)",
            "map50-95",
            "map",
        ])
        if map50 is not None and map50 > self._best_map50:
            self._best_map50 = map50
        if map5095 is not None and map5095 > self._best_map5095:
            self._best_map5095 = map5095
        self.on_progress.emit(epoch, self._best_map50, self._best_map5095)
        if self._stop_requested and not self._stop_notified:
            self._stop_notified = True
            trainer.stop_training = True
            self._log(f"Stop requested. Finishing after epoch {epoch}.")
        elif self._stop_requested:
            trainer.stop_training = True

    @staticmethod
    def _extract_metric(metrics: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
        for key, value in metrics.items():
            lower = str(key).lower()
            for probe in keys:
                normalized = probe.replace("/", "").replace("(", "").replace(")", "").lower()
                if normalized in lower:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        return None


class _SignalLogHandler(logging.Handler):
    """Logging handler to bridge Ultralytics logs into Qt signals."""

    def __init__(self, callback) -> None:
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - safety
            message = record.getMessage()
        self._callback(message)







