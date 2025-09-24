"""Main entry point for the YOLO desktop application."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QProgressDialog,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from .camera import CameraIdentifier, CameraManager, enumerate_cameras
    from .config import load_config, save_config, update_config
    from .detector import DEFAULT_MODEL, Detector
    from .ui_components import ClassSelection, ConfidenceSlider, FilePicker
    from .utils import numpy_to_qimage
    from .workers import ModelLoaderWorker, VideoWorker
except ImportError:  # pragma: no cover - support running as a script
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.camera import CameraIdentifier, CameraManager, enumerate_cameras  # type: ignore
    from app.config import load_config, save_config, update_config  # type: ignore
    from app.detector import DEFAULT_MODEL, Detector  # type: ignore
    from app.ui_components import ClassSelection, ConfidenceSlider, FilePicker  # type: ignore
    from app.utils import numpy_to_qimage  # type: ignore
    from app.workers import ModelLoaderWorker, VideoWorker  # type: ignore

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("app")

PRESET_MODELS = [
    ("YOLO11 Nano (paling cepat)", "yolo11n.pt"),
    ("YOLO11 Small (lebih akurat)", "yolo11s.pt"),
    ("YOLO11 Medium (detail lebih tinggi)", "yolo11m.pt"),
    ("YOLO11 Large (tingkat tinggi)", "yolo11l.pt"),
]
PRESET_LABELS = {path: label for label, path in PRESET_MODELS}
PRESET_PATHS = {path for _, path in PRESET_MODELS}
CUSTOM_MODEL_SENTINEL = "__custom__"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Object Detection")
        self.resize(1200, 720)

        self.config = load_config()
        self.detector = Detector()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[VideoWorker] = None
        self.camera_manager: Optional[CameraManager] = None
        self.available_cameras = enumerate_cameras()
        self._last_fps = 0.0
        self._last_inference_ms = 0.0
        self._current_frame: Optional[np.ndarray] = None

        self.model_loader_thread: Optional[QThread] = None
        self.model_loader_worker: Optional[ModelLoaderWorker] = None
        self.model_progress_dialog: Optional[QProgressDialog] = None
        self._model_loading = False
        self._pending_detection = False
        self._model_load_result: Optional[tuple[str, str]] = None

        self._build_ui()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._connect_signals()
        self._load_initial_state()

    # UI setup -----------------------------------------------------------------
    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(8)

        # Camera source controls
        self.camera_combo = QComboBox()
        self.refresh_camera_items()
        control_layout.addWidget(QLabel("Camera Devices"))
        control_layout.addWidget(self.camera_combo)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("http://<phone-ip>:8080/video")
        control_layout.addWidget(QLabel("Stream URL"))
        control_layout.addWidget(self.url_input)

        btn_row = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        btn_row.addWidget(self.connect_button)
        btn_row.addWidget(self.disconnect_button)
        control_layout.addLayout(btn_row)

        # Model controls
        control_layout.addWidget(QLabel("Model bawaan"))
        self.model_combo = QComboBox()
        for label, path_value in PRESET_MODELS:
            self.model_combo.addItem(label, path_value)
        self.model_combo.addItem("Model kustom (pilih file .pt)", CUSTOM_MODEL_SENTINEL)
        control_layout.addWidget(self.model_combo)

        control_layout.addWidget(QLabel("File model (.pt)"))
        self.model_picker = FilePicker("Select YOLO model", "PyTorch Model (*.pt)", DEFAULT_MODEL)
        control_layout.addWidget(self.model_picker)

        self.load_model_button = QPushButton("Load Model")
        control_layout.addWidget(self.load_model_button)

        self.confidence_slider = ConfidenceSlider()
        control_layout.addWidget(self.confidence_slider)

        self.class_selection = ClassSelection()
        control_layout.addWidget(self.class_selection)
        self.class_selection.setEnabled(False)

        # Detection controls
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.snapshot_button = QPushButton("Take Snapshot")
        self.record_checkbox = QCheckBox("Record annotated video")

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.snapshot_button)
        control_layout.addWidget(self.record_checkbox)
        control_layout.addStretch()

        # Log panel
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        control_layout.addWidget(QLabel("Logs"))
        control_layout.addWidget(self.log_view, stretch=1)

        # Video display
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #202020; color: #ffffff;")
        video_layout.addWidget(self.video_label)

        splitter.addWidget(control_widget)
        splitter.addWidget(video_widget)
        splitter.setStretchFactor(1, 3)

        # Menu action for refreshing devices
        refresh_action = QAction("Refresh Cameras", self)
        refresh_action.triggered.connect(self.refresh_camera_items)
        self.menuBar().addAction(refresh_action)

    def _connect_signals(self) -> None:
        self.connect_button.clicked.connect(self.on_connect)
        self.disconnect_button.clicked.connect(self.on_disconnect)
        self.start_button.clicked.connect(self.on_start_detection)
        self.stop_button.clicked.connect(self.on_stop_detection)
        self.snapshot_button.clicked.connect(self.on_snapshot)
        self.load_model_button.clicked.connect(self.on_load_model)
        self.model_combo.currentIndexChanged.connect(self.on_model_choice_changed)
        self.model_picker.valueChanged.connect(self.on_model_path_changed)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.class_selection.valueChanged.connect(self.on_class_selection_changed)
        self.record_checkbox.stateChanged.connect(self.on_record_toggle)

    def _load_initial_state(self) -> None:
        # Config -> UI
        model_choice = self.config.get("model_choice", DEFAULT_MODEL)
        self.model_combo.blockSignals(True)
        index = self.model_combo.findData(model_choice)
        if index == -1 and model_choice != CUSTOM_MODEL_SENTINEL:
            index = self.model_combo.findData(CUSTOM_MODEL_SENTINEL)
            model_choice = CUSTOM_MODEL_SENTINEL
        if index == -1:
            index = 0
            model_choice = self.model_combo.itemData(index)
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)
        self._apply_model_choice(model_choice)
        self.confidence_slider.setValue(float(self.config.get("confidence", 0.25)))
        self.url_input.setText(self.config.get("stream_url", ""))
        self.record_checkbox.setChecked(bool(self.config.get("record_enabled", False)))
        self.status_bar.showMessage("Ready")

    # Camera management --------------------------------------------------------
    def refresh_camera_items(self) -> None:
        self.available_cameras = enumerate_cameras()
        self.camera_combo.clear()
        for index, name in self.available_cameras:
            self.camera_combo.addItem(name, userData=index)
        if not self.available_cameras:
            self.camera_combo.addItem("No cameras found", userData=None)
            self.camera_combo.setEnabled(False)
        else:
            self.camera_combo.setEnabled(True)

    def on_connect(self) -> None:
        if self.worker is not None:
            QMessageBox.information(self, "Already connected", "Camera is already connected.")
            return
        source = self._determine_camera_source()
        if source is None:
            QMessageBox.warning(self, "No source", "Select a camera or enter a valid stream URL.")
            return
        self.camera_manager = CameraManager(source)
        if not self.camera_manager.open():
            QMessageBox.critical(
                self,
                "Camera Error",
                "Unable to open camera source. Check your phone app, network, or adapter.",
            )
            self.camera_manager = None
            return
        self.worker = VideoWorker(self.camera_manager, self.detector)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.statsUpdated.connect(self.on_stats_updated)
        self.worker.logMessage.connect(self.append_log)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker_thread.start()
        self.config = update_config(camera_source={"value": source, "type": "url" if isinstance(source, str) else "device"})
        self.status_bar.showMessage("Connected")
        self.append_log("Camera connected")

    def _determine_camera_source(self) -> Optional[CameraIdentifier]:
        url = self.url_input.text().strip()
        if url and url.lower().startswith("http"):
            self.config = update_config(stream_url=url)
            return url
        if not self.available_cameras:
            return None
        index = self.camera_combo.currentData()
        if index is None:
            return None
        return int(index)

    def on_disconnect(self) -> None:
        if not self.worker:
            return
        self.append_log("Disconnecting camera...")
        self.worker.request_stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker = None
        self.worker_thread = None
        self.camera_manager = None
        self.status_bar.showMessage("Disconnected")

    # Detection management ----------------------------------------------------
    def _apply_model_choice(self, choice: Optional[str], persist: bool = False) -> None:
        if choice == CUSTOM_MODEL_SENTINEL:
            self.model_picker.set_read_only(False)
            self.model_picker.set_button_enabled(True)
            stored_path = self.config.get("model_path", "")
            if stored_path in PRESET_PATHS:
                stored_path = ""
            if persist:
                self.config = update_config(model_choice=CUSTOM_MODEL_SENTINEL, model_path=stored_path)
            self.model_picker.setText(stored_path)
        else:
            effective_choice = choice or DEFAULT_MODEL
            self.model_picker.set_read_only(True)
            self.model_picker.set_button_enabled(False)
            self.model_picker.setText(effective_choice)
            if persist:
                self.config = update_config(model_choice=effective_choice, model_path=effective_choice)

    def on_model_choice_changed(self, index: int) -> None:
        choice = self.model_combo.itemData(index)
        if choice is None:
            return
        self._apply_model_choice(choice, persist=True)

    def ensure_model_loaded(self, start_detection_after: bool = False) -> bool:
        model_choice = self.model_combo.currentData()
        model_path = self.model_picker.text().strip()

        if model_choice == CUSTOM_MODEL_SENTINEL:
            if not model_path:
                QMessageBox.warning(
                    self,
                    "Model belum dipilih",
                    "Silakan pilih file model kustom (.pt) terlebih dahulu.",
                )
                return False
            choice_for_config = CUSTOM_MODEL_SENTINEL
        else:
            if not model_choice:
                model_choice = DEFAULT_MODEL
            model_path = str(model_choice)
            choice_for_config = model_path

        if (
            self.detector.names
            and self.detector.model_path
            and self.detector.model_path == model_path
        ):
            self.config = update_config(model_path=model_path, model_choice=choice_for_config)
            return True

        if self._model_loading:
            if start_detection_after:
                self._pending_detection = True
            QMessageBox.information(
                self,
                "Model sedang diproses",
                "Model sedang diunduh. Mohon tunggu hingga selesai.",
            )
            return False

        if self.worker:
            self.worker.disable_detection()
        self.detector = Detector()
        self.config = update_config(model_path=model_path, model_choice=choice_for_config)
        self._model_loading = True
        self._pending_detection = start_detection_after
        self._model_load_result = None

        self.model_progress_dialog = QProgressDialog(
            "Mengunduh dan menyiapkan model YOLO. Mohon tunggu...",
            None,
            0,
            0,
            self,
        )
        self.model_progress_dialog.setWindowTitle("Mengunduh Model")
        self.model_progress_dialog.setCancelButton(None)
        self.model_progress_dialog.setWindowModality(Qt.WindowModal)
        self.model_progress_dialog.setMinimumDuration(0)
        self.model_progress_dialog.show()
        QApplication.processEvents()

        self.model_loader_thread = QThread()
        self.model_loader_worker = ModelLoaderWorker(model_path)
        self.model_loader_worker.moveToThread(self.model_loader_thread)
        self.model_loader_thread.started.connect(self.model_loader_worker.run)
        self.model_loader_worker.modelLoaded.connect(self.on_model_loaded)
        self.model_loader_worker.error.connect(self.on_model_load_error)
        self.model_loader_worker.finished.connect(self.on_model_load_finished)
        self.model_loader_thread.start()
        return False

    def on_load_model(self) -> None:
        self.ensure_model_loaded()

    def on_start_detection(self) -> None:
        if not self.worker:
            QMessageBox.warning(self, "No camera", "Connect to a camera before starting detection.")
            return
        if not self.detector.names:
            self.ensure_model_loaded(start_detection_after=True)
            return
        self._start_detection_now()

    def _start_detection_now(self) -> None:
        if not self.worker:
            return
        selected_labels = self.class_selection.selected()
        class_ids = self.detector.class_ids_from_labels(selected_labels) if selected_labels else []
        confidence = self.confidence_slider.value()
        record = self.record_checkbox.isChecked()
        self.worker.enable_detection(confidence, class_ids, record)
        self.config = update_config(
            confidence=confidence,
            selected_classes=selected_labels,
            record_enabled=record,
        )
        self.append_log("Detection started")
        self.status_bar.showMessage("Detection running")

    def on_stop_detection(self) -> None:
        if not self.worker:
            return
        self.worker.disable_detection()
        self.status_bar.showMessage("Detection stopped")

    def on_model_loaded(self, detector: Detector, classes: List[str], model_path: str) -> None:
        self.detector = detector
        self.class_selection.setEnabled(True)
        self.class_selection.set_classes(classes)
        saved_selection = self.config.get("selected_classes") or []
        if saved_selection:
            self._apply_class_selection(saved_selection)
        friendly_name = PRESET_LABELS.get(model_path, Path(model_path).name)
        self.append_log(f"Model dimuat: {friendly_name} ({model_path})")
        self.status_bar.showMessage("Model berhasil dimuat")
        if model_path in PRESET_PATHS:
            self.model_combo.blockSignals(True)
            index = self.model_combo.findData(model_path)
            if index != -1:
                self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)
            self._apply_model_choice(model_path, persist=True)
        else:
            custom_index = self.model_combo.findData(CUSTOM_MODEL_SENTINEL)
            self.model_combo.blockSignals(True)
            if custom_index != -1:
                self.model_combo.setCurrentIndex(custom_index)
            self.model_combo.blockSignals(False)
            self._apply_model_choice(CUSTOM_MODEL_SENTINEL, persist=True)
            self.model_picker.setText(model_path)
            self.config = update_config(model_path=model_path, model_choice=CUSTOM_MODEL_SENTINEL)
        if self.worker:
            self.worker.update_detector(self.detector)
        self._model_load_result = (
            "success",
            f"Unduhan model selesai: {friendly_name}. Model siap digunakan.",
        )
        if self._pending_detection:
            self._pending_detection = False
            self._start_detection_now()

    def on_model_load_error(self, message: str) -> None:
        self.append_log(f"Model load error: {message}")
        self.status_bar.showMessage("Gagal memuat model")
        self._model_load_result = (
            "error",
            f"Gagal mengunduh atau memuat model. Detail: {message}",
        )
        self._pending_detection = False

    def on_model_load_finished(self) -> None:
        self._model_loading = False
        if self.model_progress_dialog:
            self.model_progress_dialog.close()
            self.model_progress_dialog = None
        if self.model_loader_worker:
            self.model_loader_worker.deleteLater()
            self.model_loader_worker = None
        if self.model_loader_thread:
            self.model_loader_thread.quit()
            self.model_loader_thread.wait()
            self.model_loader_thread = None
        if self._model_load_result:
            status, message = self._model_load_result
            if status == "success":
                QMessageBox.information(self, "Model siap", message)
            else:
                QMessageBox.critical(self, "Model gagal dimuat", message)
            self._model_load_result = None

    def on_snapshot(self) -> None:
        if not self.worker:
            QMessageBox.information(self, "Not connected", "Connect to a camera first.")
            return
        self.worker.request_snapshot()
        self.append_log("Snapshot requested")

    # Signal handlers ---------------------------------------------------------
    def on_frame_ready(self, frame: np.ndarray) -> None:
        self._current_frame = frame
        annotated = frame.copy()
        overlay_lines = [f"FPS: {self._last_fps:.1f}"]
        if self._last_inference_ms:
            overlay_lines.append(f"Inference: {self._last_inference_ms:.1f} ms")
        y_offset = 30
        for line in overlay_lines:
            cv2.putText(annotated, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y_offset += 30
        image = numpy_to_qimage(annotated)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.video_label.pixmap():
            self.video_label.setPixmap(
                self.video_label.pixmap().scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def on_stats_updated(self, fps: float, inference_ms: float) -> None:
        self._last_fps = fps
        self._last_inference_ms = inference_ms
        self.status_bar.showMessage(f"FPS: {fps:.1f} | Inference: {inference_ms:.1f} ms")

    def on_worker_error(self, message: str) -> None:
        self.append_log(f"Error: {message}")
        self.status_bar.showMessage(message)

    def on_worker_finished(self) -> None:
        self.append_log("Worker finished")
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker = None
        self.worker_thread = None
        self.camera_manager = None

    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    # Config updates ----------------------------------------------------------
    def _apply_class_selection(self, classes: List[str]) -> None:
        for index in range(self.class_selection.list_widget.count()):
            item = self.class_selection.list_widget.item(index)
            item.setCheckState(Qt.Checked if item.text() in classes else Qt.Unchecked)

    def on_model_path_changed(self, path: str) -> None:
        trimmed = path.strip()
        self.config = update_config(model_path=trimmed)
        if trimmed in PRESET_PATHS and self.model_combo.currentData() != trimmed:
            index = self.model_combo.findData(trimmed)
            if index != -1:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(index)
                self.model_combo.blockSignals(False)
                self._apply_model_choice(trimmed, persist=True)

    def on_confidence_changed(self, value: float) -> None:
        self.config = update_config(confidence=value)

    def on_class_selection_changed(self, classes: List[str]) -> None:
        self.config = update_config(selected_classes=classes)

    def on_record_toggle(self, _state: int) -> None:
        self.config = update_config(record_enabled=self.record_checkbox.isChecked())

    # Shutdown ----------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.append_log("Closing application")
        if self.worker:
            self.worker.disable_detection()
            self.worker.request_stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)
        if self.camera_manager:
            self.camera_manager.release()
        save_config(self.config)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())








