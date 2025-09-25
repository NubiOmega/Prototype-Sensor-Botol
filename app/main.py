"""Main entry point for the YOLO desktop application."""
from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressDialog,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - optional torch dependency
    torch = None

try:
    from .camera import CameraIdentifier, CameraManager, enumerate_cameras
    from .config import DEFAULT_TRAINING, load_config, save_config, update_config
    from .detector import DEFAULT_MODEL, Detector
    from .ui_components import ClassSelection, ConfidenceSlider, FilePicker
    from .utils import numpy_to_qimage, timestamp_for_file
    from .workers import ModelLoaderWorker, TrainingWorker, VideoWorker
except ImportError:  # pragma: no cover - support running as a script
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.camera import CameraIdentifier, CameraManager, enumerate_cameras  # type: ignore
    from app.config import DEFAULT_TRAINING, load_config, save_config, update_config  # type: ignore
    from app.detector import DEFAULT_MODEL, Detector  # type: ignore
    from app.ui_components import ClassSelection, ConfidenceSlider, FilePicker  # type: ignore
    from app.utils import numpy_to_qimage, timestamp_for_file  # type: ignore
    from app.workers import ModelLoaderWorker, TrainingWorker, VideoWorker  # type: ignore

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
        self.training_worker: Optional[TrainingWorker] = None
        self.training_log_path: Optional[Path] = None
        self.training_best_weights: Optional[Path] = None
        self._training_stop_requested = False
        self._loading_training_settings = False
        self._cuda_available = bool(torch and hasattr(torch, "cuda") and torch.cuda.is_available())

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

        self.tab_widget = QTabWidget()
        detection_widget = self._build_detection_tab()
        training_widget = self._build_training_tab()
        self.tab_widget.addTab(detection_widget, "Detection")
        self.tab_widget.addTab(training_widget, "Training")

        splitter.addWidget(self.tab_widget)
        splitter.addWidget(self._build_video_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        refresh_action = QAction("Refresh Cameras", self)
        refresh_action.triggered.connect(self.refresh_camera_items)
        self.menuBar().addAction(refresh_action)

    def _build_detection_tab(self) -> QWidget:
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(8)

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

        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.snapshot_button = QPushButton("Take Snapshot")
        self.record_checkbox = QCheckBox("Record annotated video")

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.snapshot_button)
        control_layout.addWidget(self.record_checkbox)
        control_layout.addStretch()

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        control_layout.addWidget(QLabel("Logs"))
        control_layout.addWidget(self.log_view, stretch=1)

        return control_widget

    def _build_training_tab(self) -> QWidget:
        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.training_data_picker = FilePicker("Select dataset YAML", "YAML files (*.yaml *.yml)", DEFAULT_TRAINING["data_path"])
        form.addRow("Dataset YAML", self.training_data_picker)

        self.training_model_picker = FilePicker("Select base model", "PyTorch Model (*.pt)", DEFAULT_TRAINING["model_path"])
        form.addRow("Base Model", self.training_model_picker)

        self.training_epochs_spin = QSpinBox()
        self.training_epochs_spin.setRange(10, 300)
        self.training_epochs_spin.setValue(int(DEFAULT_TRAINING["epochs"]))
        form.addRow("Epochs", self.training_epochs_spin)

        self.training_imgsz_spin = QSpinBox()
        self.training_imgsz_spin.setRange(640, 1280)
        self.training_imgsz_spin.setSingleStep(32)
        self.training_imgsz_spin.setValue(int(DEFAULT_TRAINING["imgsz"]))
        form.addRow("Image size", self.training_imgsz_spin)

        self.training_batch_spin = QSpinBox()
        self.training_batch_spin.setRange(4, 32)
        self.training_batch_spin.setValue(int(DEFAULT_TRAINING["batch"]))
        form.addRow("Batch size", self.training_batch_spin)

        self.training_workers_spin = QSpinBox()
        self.training_workers_spin.setRange(0, 4)
        self.training_workers_spin.setValue(int(DEFAULT_TRAINING["workers"]))
        form.addRow("Dataloader workers", self.training_workers_spin)

        self.training_device_combo = QComboBox()
        self._populate_device_options(DEFAULT_TRAINING["device"])
        form.addRow("Device", self.training_device_combo)

        layout.addLayout(form)

        checkbox_row = QHBoxLayout()
        self.training_resume_checkbox = QCheckBox("Resume from last checkpoint")
        self.training_noval_checkbox = QCheckBox("Skip validation (faster)")
        checkbox_row.addWidget(self.training_resume_checkbox)
        checkbox_row.addWidget(self.training_noval_checkbox)
        checkbox_row.addStretch()
        layout.addLayout(checkbox_row)

        button_row = QHBoxLayout()
        self.training_start_button = QPushButton("Start Training")
        self.training_stop_button = QPushButton("Stop Training")
        self.training_stop_button.setEnabled(False)
        button_row.addWidget(self.training_start_button)
        button_row.addWidget(self.training_stop_button)
        layout.addLayout(button_row)

        layout.addWidget(QLabel("Training Logs"))
        self.training_log_view = QPlainTextEdit()
        self.training_log_view.setReadOnly(True)
        self.training_log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.training_log_view, stretch=1)

        metrics_row = QHBoxLayout()
        self.training_epoch_label = QLabel("Epoch: -")
        self.training_map50_label = QLabel("Best mAP50: -")
        self.training_map5095_label = QLabel("Best mAP50-95: -")
        metrics_row.addWidget(self.training_epoch_label)
        metrics_row.addWidget(self.training_map50_label)
        metrics_row.addWidget(self.training_map5095_label)
        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        self.training_load_model_button = QPushButton("Load Trained Model")
        self.training_load_model_button.setEnabled(False)
        layout.addWidget(self.training_load_model_button)

        return training_widget

    def _build_video_panel(self) -> QWidget:
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #202020; color: #ffffff;")
        video_layout.addWidget(self.video_label)
        return video_widget
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
        self.training_start_button.clicked.connect(self.on_start_training)
        self.training_stop_button.clicked.connect(self.on_stop_training)
        self.training_load_model_button.clicked.connect(self.on_load_trained_model)
        self.training_data_picker.valueChanged.connect(self.on_training_settings_changed)
        self.training_model_picker.valueChanged.connect(self.on_training_settings_changed)
        self.training_epochs_spin.valueChanged.connect(self.on_training_settings_changed)
        self.training_imgsz_spin.valueChanged.connect(self.on_training_settings_changed)
        self.training_batch_spin.valueChanged.connect(self.on_training_settings_changed)
        self.training_workers_spin.valueChanged.connect(self.on_training_settings_changed)
        self.training_device_combo.currentIndexChanged.connect(self.on_training_settings_changed)
        self.training_resume_checkbox.stateChanged.connect(self.on_training_settings_changed)
        self.training_noval_checkbox.stateChanged.connect(self.on_training_settings_changed)
    def _load_initial_state(self) -> None:
        model_choice = self.config.get("model_choice", DEFAULT_MODEL)
        self.model_combo.blockSignals(True)
        index = self.model_combo.findData(model_choice)
        if index == -1 and model_choice != CUSTOM_MODEL_SENTINEL:
            index = self.model_combo.findData(CUSTOM_MODEL_SENTINEL)
        if index != -1:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

        stored_model_path = self.config.get("model_path", DEFAULT_MODEL)
        if stored_model_path:
            self.model_picker.setText(stored_model_path)
        else:
            self.model_picker.setText(DEFAULT_MODEL)

        confidence = float(self.config.get("confidence", 0.25))
        self.confidence_slider.setValue(confidence)

        selected_classes = self.config.get("selected_classes", [])
        if isinstance(selected_classes, list):
            self._apply_class_selection(selected_classes)

        self.record_checkbox.setChecked(bool(self.config.get("record_enabled", False)))

        training_cfg = self.config.get("training", DEFAULT_TRAINING)
        self._loading_training_settings = True
        self.training_data_picker.setText(str(training_cfg.get("data_path", DEFAULT_TRAINING["data_path"])))
        self.training_model_picker.setText(str(training_cfg.get("model_path", DEFAULT_TRAINING["model_path"])))
        self.training_epochs_spin.setValue(int(training_cfg.get("epochs", DEFAULT_TRAINING["epochs"])))
        self.training_imgsz_spin.setValue(int(training_cfg.get("imgsz", DEFAULT_TRAINING["imgsz"])))
        self.training_batch_spin.setValue(int(training_cfg.get("batch", DEFAULT_TRAINING["batch"])))
        self.training_workers_spin.setValue(int(training_cfg.get("workers", DEFAULT_TRAINING["workers"])))
        self._populate_device_options(str(training_cfg.get("device", DEFAULT_TRAINING["device"])))
        self.training_resume_checkbox.setChecked(bool(training_cfg.get("resume", DEFAULT_TRAINING["resume"])))
        self.training_noval_checkbox.setChecked(bool(training_cfg.get("noval", DEFAULT_TRAINING["noval"])))
        self._loading_training_settings = False

        self.training_log_view.clear()
        self.training_epoch_label.setText("Epoch: -")
        self.training_map50_label.setText("Best mAP50: -")
        self.training_map5095_label.setText("Best mAP50-95: -")

        last_weights = Path(self.config.get("last_trained_weights", ""))
        if last_weights.exists():
            self.training_best_weights = last_weights
            self.training_load_model_button.setEnabled(True)
        else:
            self.training_best_weights = None
            self.training_load_model_button.setEnabled(False)
    def _populate_device_options(self, selected: str) -> None:
        self.training_device_combo.clear()
        self.training_device_combo.addItem("CPU", "cpu")
        if self._cuda_available:
            self.training_device_combo.addItem("CUDA:0", "cuda:0")
        index = self.training_device_combo.findData(selected)
        if index == -1:
            index = 0
        self.training_device_combo.setCurrentIndex(index)

    def _collect_training_settings(self) -> Dict[str, object]:
        data_path = self.training_data_picker.text().strip() or DEFAULT_TRAINING["data_path"]
        model_path = self.training_model_picker.text().strip() or DEFAULT_TRAINING["model_path"]
        device_data = self.training_device_combo.currentData()
        device = device_data if isinstance(device_data, str) else self.training_device_combo.currentText()
        return {
            "data_path": data_path,
            "model_path": model_path,
            "epochs": int(self.training_epochs_spin.value()),
            "imgsz": int(self.training_imgsz_spin.value()),
            "batch": int(self.training_batch_spin.value()),
            "workers": int(self.training_workers_spin.value()),
            "device": device,
            "resume": bool(self.training_resume_checkbox.isChecked()),
            "noval": bool(self.training_noval_checkbox.isChecked()),
        }

    def _persist_training_settings(self) -> None:
        settings = self._collect_training_settings()
        self.config = update_config(training=settings)

    def on_training_settings_changed(self, *_) -> None:
        if self._loading_training_settings:
            return
        self._persist_training_settings()

    def _validate_dataset(self, yaml_path: str) -> tuple[bool, str]:
        config_path = Path(yaml_path)
        if not config_path.exists():
            return False, f"Dataset config not found: {config_path}"
        try:
            import dataset_tools  # type: ignore
        except ImportError:
            return True, ""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ok = dataset_tools.check_dataset(str(config_path))
        message = buffer.getvalue().strip()
        return ok, message

    def append_training_log(self, message: str) -> None:
        self.training_log_view.appendPlainText(message)
        scrollbar = self.training_log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_start_training(self) -> None:
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.information(self, "Training running", "Training is already in progress.")
            return

        settings = self._collect_training_settings()
        data_path = Path(settings["data_path"])
        if not data_path.exists():
            QMessageBox.warning(self, "Dataset missing", f"Dataset YAML not found: {data_path}")
            return

        dataset_ok, dataset_message = self._validate_dataset(str(data_path))
        if dataset_message:
            self.append_training_log(dataset_message)
        if not dataset_ok:
            QMessageBox.warning(self, "Invalid dataset", dataset_message or "Dataset validation failed.")
            return

        model_path = Path(settings["model_path"])
        if not model_path.exists():
            QMessageBox.warning(self, "Model missing", f"Base model not found: {model_path}")
            return

        params = {
            "model": str(model_path),
            "data": str(data_path),
            "epochs": int(settings["epochs"]),
            "imgsz": int(settings["imgsz"]),
            "batch": int(settings["batch"]),
            "device": str(settings["device"]),
            "workers": int(settings["workers"]),
            "resume": bool(settings["resume"]),
            "noval": bool(settings["noval"]),
            "project": "runs/detect",
            "name": "train",
        }

        self._persist_training_settings()
        self.training_log_view.clear()
        self.training_epoch_label.setText("Epoch: 0")
        self.training_map50_label.setText("Best mAP50: -")
        self.training_map5095_label.setText("Best mAP50-95: -")

        self.training_log_path = Path("runs") / f"training_{timestamp_for_file()}.log"
        self.append_training_log(f"Logging to {self.training_log_path}")

        self.training_worker = TrainingWorker(params, self.training_log_path)
        self.training_worker.on_log.connect(self.append_training_log)
        self.training_worker.on_progress.connect(self.on_training_progress)
        self.training_worker.on_finished.connect(self.on_training_finished)
        self.training_worker.on_error.connect(self.on_training_error)

        self.training_start_button.setEnabled(False)
        self.training_stop_button.setEnabled(True)
        self._training_stop_requested = False
        self.training_worker.start()
        self.status_bar.showMessage("Training started", 5000)

    def on_stop_training(self) -> None:
        if not self.training_worker or not self.training_worker.isRunning():
            QMessageBox.information(self, "Training", "No training in progress.")
            return
        self._training_stop_requested = True
        self.training_worker.request_stop()
        self.training_stop_button.setEnabled(False)
        self.append_training_log("Stop requested. Waiting for epoch to finish...")

    def on_training_progress(self, epoch: int, best_map50: float, best_map5095: float) -> None:
        self.training_epoch_label.setText(f"Epoch: {epoch}")
        map50_text = "Best mAP50: -" if best_map50 <= 0 else f"Best mAP50: {best_map50:.4f}"
        map5095_text = "Best mAP50-95: -" if best_map5095 <= 0 else f"Best mAP50-95: {best_map5095:.4f}"
        self.training_map50_label.setText(map50_text)
        self.training_map5095_label.setText(map5095_text)
        self.status_bar.showMessage(f"Training epoch {epoch}", 2000)

    def on_training_finished(self, best_weights_path: str) -> None:
        if self.training_worker:
            self.training_worker.wait(200)
            self.training_worker.deleteLater()
            self.training_worker = None
        self.training_start_button.setEnabled(True)
        self.training_stop_button.setEnabled(False)
        best_path = Path(best_weights_path)
        if best_path.exists():
            self.training_best_weights = best_path
            self.training_load_model_button.setEnabled(True)
            self.config = update_config(last_trained_weights=str(best_path))
            self.append_training_log(f"Training finished. Best weights at {best_path}")
            self.status_bar.showMessage("Training completed", 5000)
        else:
            self.append_training_log(f"Training finished, but weights not found at {best_path}")
            self.status_bar.showMessage("Training finished (weights missing)", 5000)
            self.training_best_weights = None
            self.training_load_model_button.setEnabled(False)
        if self._training_stop_requested:
            self.append_training_log("Training stopped by user.")
        self._training_stop_requested = False

    def on_training_error(self, message: str) -> None:
        if self.training_worker:
            self.training_worker.deleteLater()
            self.training_worker = None
        self.training_start_button.setEnabled(True)
        self.training_stop_button.setEnabled(False)
        self.append_training_log(f"Error: {message}")
        QMessageBox.critical(self, "Training failed", message)

    def on_load_trained_model(self) -> None:
        path = self.training_best_weights or Path(self.config.get("last_trained_weights", ""))
        if not path or not path.exists():
            QMessageBox.warning(self, "Model not found", "No trained model available to load.")
            return
        path = path.resolve()
        self.append_training_log(f"Loading trained model: {path}")
        index = self.model_combo.findData(CUSTOM_MODEL_SENTINEL)
        if index != -1:
            self.model_combo.blockSignals(True)
            self.model_combo.setCurrentIndex(index)
            self.model_combo.blockSignals(False)
        self.model_picker.setText(str(path))
        self.config = update_config(model_choice=CUSTOM_MODEL_SENTINEL, model_path=str(path), last_trained_weights=str(path))
        self.ensure_model_loaded()
        QMessageBox.information(self, "Model loaded", "Trained model loaded into detection tab.")
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
        if self.training_worker and self.training_worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Training in progress",
                "Training is running. Stop training and exit?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.on_stop_training()
            if self.training_worker:
                self.training_worker.wait(2000)

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









