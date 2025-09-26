"""Worker threads for capture, detection, and training."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread
from sqlalchemy import func, select

from ultralytics import YOLO

from .camera import CameraManager
from .db import (
    Detection,
    DetectionPayload,
    Inspection,
    batch_counters,
    close_batch,
    record_inspection,
    session_scope,
    upsert_batch,
)
from .detector import Detector, DetectionResult
from .qc_rules import RuleDefinition
from .roi import ROIShape, roi_any_contains
from .utils import (
    FpsCalculator,
    INSPECTION_DIR,
    RECORD_DIR,
    SNAPSHOT_DIR,
    ensure_dirs,
    save_frame,
    timestamp_for_file,
)

try:  # pragma: no cover - optional dependency at runtime
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchContext:
    lot_id: str
    line: str
    shift: str
    operator: str
    notes: str = ""


class VideoWorker(QObject):
    frameReady = Signal(np.ndarray)
    statsUpdated = Signal(float, float)
    logMessage = Signal(str)
    error = Signal(str)
    finished = Signal()
    inspectionRecorded = Signal(dict)
    countersUpdated = Signal(dict)
    batchChanged = Signal(dict)

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
        self._batch_context: Optional[BatchContext] = None
        self._batch_id: Optional[int] = None
        self._roi_shapes: List[ROIShape] = []
        self._rule: Optional[RuleDefinition] = None
        self._model_id: Optional[int] = None
        self._recent_detections: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._pending_note: str = ""
        self._next_override: Optional[str] = None
        self._auto_snapshot_fail = True

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

    def enable_detection(
        self,
        confidence: float,
        class_ids: Iterable[int],
        record: bool,
        *,
        batch: Optional[BatchContext] = None,
        rule: Optional[RuleDefinition] = None,
        roi: Optional[Sequence[ROIShape]] = None,
        model_id: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._confidence = confidence
            self._class_ids = list(class_ids)
            self._detection_enabled = True
            self._record_enabled = record
            self._batch_context = batch
            self._rule = rule
            self._roi_shapes = list(roi or [])
            self._model_id = model_id
            self._pending_note = ""
            self._next_override = None
            self._batch_id = None
            self._recent_detections.clear()
        classes_display = self._class_ids or "all"
        self.logMessage.emit(
            f"Detection enabled (conf={confidence:.2f}, classes={classes_display}, record={record})"
        )

    def disable_detection(self) -> None:
        with self._lock:
            self._detection_enabled = False
            self._record_enabled = False
        self._tear_down_writer()
        self._close_active_batch()
        self.logMessage.emit("Detection disabled")

    def request_snapshot(self) -> None:
        with self._lock:
            self._pending_snapshot = True

    def update_detector(self, detector: Detector) -> None:
        with self._lock:
            self.detector = detector

    def update_rule(self, rule: Optional[RuleDefinition]) -> None:
        with self._lock:
            self._rule = rule

    def update_roi(self, roi: Sequence[ROIShape]) -> None:
        with self._lock:
            self._roi_shapes = list(roi)

    def update_batch(self, batch: Optional[BatchContext]) -> None:
        with self._lock:
            self._batch_context = batch
            if batch is None:
                self._batch_id = None

    def update_model_id(self, model_id: Optional[int]) -> None:
        with self._lock:
            self._model_id = model_id

    def queue_note(self, note: str) -> None:
        with self._lock:
            self._pending_note = note

    def override_next(self, status: Optional[str]) -> None:
        normalized = status.upper() if status else None
        if normalized not in {"PASS", "FAIL", None}:
            raise ValueError("Override status must be PASS, FAIL, or None")
        with self._lock:
            self._next_override = normalized

    def set_auto_snapshot_fail(self, enabled: bool) -> None:
        with self._lock:
            self._auto_snapshot_fail = enabled

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
                roi_shapes = list(self._roi_shapes)
                rule = self._rule
                batch_context = self._batch_context
                model_id = self._model_id
                note = self._pending_note
                override = self._next_override
                auto_snapshot_fail = self._auto_snapshot_fail
            if not running:
                break

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            detections: List[DetectionResult] = []
            annotated = frame
            if detection_enabled and detector:
                try:
                    annotated, detections, inference_ms = detector.predict(frame, confidence, class_ids or None)
                except Exception as exc:  # pragma: no cover - runtime safety
                    LOGGER.exception("Detector failed")
                    self.error.emit(f"Detection error: {exc}")
                    self.disable_detection()
                    detection_enabled = False
                    inference_ms = 0.0
            else:
                inference_ms = 0.0

            filtered = self._filter_detections(detections, roi_shapes)

            fps = self._fps.update()
            self.frameReady.emit(annotated)
            self.statsUpdated.emit(fps, inference_ms)

            recorded = False
            if detection_enabled:
                recorded = self._handle_inspection(
                    original_frame=frame,
                    annotated_frame=annotated,
                    detections=filtered,
                    fps=fps,
                    inference_ms=inference_ms,
                    batch_context=batch_context,
                    model_id=model_id,
                    rule=rule,
                    note=note,
                    override_status=override,
                    auto_snapshot_fail=auto_snapshot_fail,
                )

            if record_enabled:
                self._write_frame(annotated)
            if pending_snapshot:
                self._save_snapshot(annotated)
                with self._lock:
                    self._pending_snapshot = False

            if recorded:
                with self._lock:
                    self._pending_note = ""
                    self._next_override = None

            self._iterations += 1
            if self._max_iterations and self._iterations >= self._max_iterations:
                break
        self._finalize()

    def _filter_detections(
        self,
        detections: Sequence[DetectionResult],
        roi_shapes: Sequence[ROIShape],
    ) -> List[DetectionResult]:
        if not roi_shapes:
            return list(detections)
        filtered: List[DetectionResult] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox:
                continue
            if roi_any_contains(roi_shapes, tuple(bbox)):  # type: ignore[arg-type]
                filtered.append(det)
        return filtered

    def _handle_inspection(
        self,
        *,
        original_frame: np.ndarray,
        annotated_frame: np.ndarray,
        detections: Sequence[DetectionResult],
        fps: float,
        inference_ms: float,
        batch_context: Optional[BatchContext],
        model_id: Optional[int],
        rule: Optional[RuleDefinition],
        note: Optional[str],
        override_status: Optional[str],
        auto_snapshot_fail: bool,
    ) -> bool:
        status = self._resolve_status(detections, rule, override_status)
        payloads = self._convert_payloads(detections)
        frame_path = ""
        if auto_snapshot_fail and status == "FAIL":
            filename = INSPECTION_DIR / f"fail_{timestamp_for_file()}.jpg"
            save_frame(annotated_frame, filename)
            frame_path = str(filename)
            self.logMessage.emit(f"Fail snapshot saved to {filename}")

        with session_scope() as session:
            batch_id = None
            batch_opened = False
            if batch_context:
                batch = upsert_batch(
                    session,
                    lot_id=batch_context.lot_id,
                    line=batch_context.line,
                    shift=batch_context.shift,
                    operator=batch_context.operator,
                    notes=batch_context.notes,
                )
                batch_id = batch.id
                if self._batch_id != batch_id:
                    batch_opened = True
            inspection = record_inspection(
                session,
                batch_id=batch_id,
                frame_path=frame_path,
                pass_fail=status,
                rule_version=rule.version if rule else "",
                inference_ms=inference_ms,
                fps=fps,
                model_id=model_id,
                detections=payloads,
                notes=note or "",
            )
            class_counts: Dict[str, int] = {}
            pass_count = fail_count = 0
            if batch_id is not None:
                totals = session.execute(
                    select(Inspection.pass_fail, func.count(Inspection.id))
                    .where(Inspection.batch_id == batch_id)
                    .group_by(Inspection.pass_fail)
                ).all()
                for key, value in totals:
                    if key == "PASS":
                        pass_count = int(value)
                    elif key == "FAIL":
                        fail_count = int(value)
                class_counts = batch_counters(session, batch_id)

        with self._lock:
            if batch_id is not None:
                self._batch_id = batch_id
            event = {
                "inspection_id": inspection.id,
                "created_at": inspection.created_at.isoformat() if inspection.created_at else datetime.utcnow().isoformat(),
                "pass_fail": status,
                "detections": [
                    {
                        "label": det.get("label"),
                        "confidence": float(det.get("confidence", 0.0) or 0.0),
                        "bbox": det.get("bbox"),
                    }
                    for det in detections
                ],
                "frame_path": frame_path,
                "batch_id": batch_id,
                "note": note or "",
                "rule_version": rule.version if rule else "",
            }
            self._recent_detections.appendleft(event)
        self.inspectionRecorded.emit(event)
        self.countersUpdated.emit(
            {
                "batch_id": batch_id,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "defect_counts": class_counts,
                "recent": list(self._recent_detections),
            }
        )
        if batch_opened:
            self.batchChanged.emit(
                {
                    "status": "opened",
                    "batch_id": batch_id,
                    "lot_id": batch_context.lot_id if batch_context else "",
                    "line": batch_context.line if batch_context else "",
                }
            )
        return True

    def _resolve_status(
        self,
        detections: Sequence[DetectionResult],
        rule: Optional[RuleDefinition],
        override_status: Optional[str],
    ) -> str:
        if override_status:
            return override_status
        if not rule:
            return "PASS"
        return rule.evaluate(detections)

    def _convert_payloads(self, detections: Sequence[DetectionResult]) -> List[DetectionPayload]:
        payloads: List[DetectionPayload] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            payloads.append(
                DetectionPayload(
                    cls=str(det.get("label", "")),
                    conf=float(det.get("confidence", 0.0) or 0.0),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return payloads

    def _save_snapshot(self, frame: np.ndarray) -> None:
        filename = SNAPSHOT_DIR / f"snapshot_{timestamp_for_file()}.jpg"
        save_frame(frame, filename)
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
        fps_value = max(1.0, self._fps.current_fps or 15.0)
        writer = cv2.VideoWriter(str(filename), fourcc, fps_value, (width, height))
        if not writer.isOpened():
            self.logMessage.emit("Falling back to AVI writer")
            filename = RECORD_DIR / f"record_{timestamp_for_file()}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(filename), fourcc, fps_value, (width, height))
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

    def _close_active_batch(self) -> None:
        batch_id = self._batch_id
        if not batch_id:
            return
        with session_scope() as session:
            close_batch(session, batch_id)
        self.batchChanged.emit({"status": "closed", "batch_id": batch_id})
        self._batch_id = None

    def _finalize(self) -> None:
        self._tear_down_writer()
        self._close_active_batch()
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
        self._log_callback: Callable[[str], None] | None = None
        self._stop_requested = False
        self._stop_notified = False
        self._best_map50 = 0.0
        self._best_map5095 = 0.0

    def request_stop(self) -> None:
        self._stop_requested = True

    def set_log_callback(self, callback: Callable[[str], None]) -> None:
        self._log_callback = callback

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("Training worker failed")
            self.on_error.emit(str(exc))

    def _execute(self) -> None:
        params = dict(self.params)
        model_path = params.pop("model_path")
        project = Path(params.pop("project", "runs/train"))
        name = params.pop("name", timestamp_for_file())
        allow_resume = params.pop("resume", False)
        noval = params.pop("noval", False)
        workers = params.pop("workers", 0)
        device = params.pop("device", "cpu")

        project.mkdir(parents=True, exist_ok=True)
        self._log(f"Training output -> {project / name}")

        model = YOLO(model_path)
        train_kwargs = {
            "project": str(project),
            "name": name,
            "exist_ok": allow_resume,
            "device": device,
            "workers": workers,
            "noval": noval,
        }
        train_kwargs.update(params)

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


__all__ = [
    "BatchContext",
    "ModelLoaderWorker",
    "TrainingWorker",
    "VideoWorker",
]




