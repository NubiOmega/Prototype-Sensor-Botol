"""Main entry point for the YOLO desktop application."""
from __future__ import annotations

import io
import json
import logging
import sys
from collections import deque
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QDate, Qt, QThread, QUrl, QRectF, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressDialog,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - optional torch dependency
    import torch
except Exception:  # pragma: no cover - optional torch dependency
    torch = None

try:
    from .analytics import AnalyticsFilters, AnalyticsResult, run_analytics
    from .camera import CameraIdentifier, CameraManager, enumerate_cameras
    from .colab_gen import ColabConfig, generate_notebook
    from .config import DEFAULT_TRAINING, load_config, save_config, update_config
    from .db import init_db, session_scope, Detection, Inspection, ModelRecord, ReviewTruth, RoiPreset
    from .detector import DEFAULT_MODEL, Detector
    from .exporters import export_csv, export_pdf
    from .models_registry import (
        ModelInfo,
        active_model_info,
        list_model_info,
        prune_missing_models,
        register_model_path,
        set_active,
    )
    from .qc_rules import RuleDefinition, ensure_rule_exists, list_rules, save_rule, set_active_rule
    from .reports import ReportMetadata
    from .dataset_tools import ReviewedItem, LabelBox, export_reviewed_items
    from .review import ReviewWidget, DetectionBox
    from .roi import ROIManager, ROIShape
    from .ui_components import ClassSelection, ConfidenceSlider, FilePicker
    from .utils import numpy_to_qimage, timestamp_for_file
    from .workers import BatchContext, ModelLoaderWorker, TrainingWorker, VideoWorker
except ImportError:  # pragma: no cover - support running as a script
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.analytics import AnalyticsFilters, AnalyticsResult, run_analytics  # type: ignore
    from app.camera import CameraIdentifier, CameraManager, enumerate_cameras  # type: ignore
    from app.colab_gen import ColabConfig, generate_notebook  # type: ignore
    from app.config import DEFAULT_TRAINING, load_config, save_config, update_config  # type: ignore
    from app.db import init_db, session_scope, Detection, Inspection, ModelRecord, ReviewTruth, RoiPreset  # type: ignore
    from app.detector import DEFAULT_MODEL, Detector  # type: ignore
    from app.exporters import export_csv, export_pdf  # type: ignore
    from app.models_registry import (  # type: ignore
        ModelInfo,
        active_model_info,
        list_model_info,
        prune_missing_models,
        register_model_path,
        set_active,
    )
    from app.qc_rules import RuleDefinition, ensure_rule_exists, list_rules, save_rule, set_active_rule  # type: ignore
    from app.reports import ReportMetadata  # type: ignore
    from app.dataset_tools import ReviewedItem, LabelBox, export_reviewed_items  # type: ignore
    from app.review import ReviewWidget, DetectionBox  # type: ignore
    from app.roi import ROIManager, ROIShape  # type: ignore
    from app.ui_components import ClassSelection, ConfidenceSlider, FilePicker  # type: ignore
    from app.utils import numpy_to_qimage, timestamp_for_file  # type: ignore
    from app.workers import BatchContext, ModelLoaderWorker, TrainingWorker, VideoWorker  # type: ignore
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





class CameraPopupWindow(QWidget):
    closed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Camera Preview")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumSize(640, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.label = QLabel("Camera feed will appear here")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #202020; color: #ffffff;")
        layout.addWidget(self.label)

        self._pixmap: Optional[QPixmap] = None

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._update_display()

    def clear(self) -> None:
        self._pixmap = None
        self.label.clear()
        self.label.setText("Camera feed will appear here")

    def _update_display(self) -> None:
        if not self._pixmap or self.label.width() <= 0 or self.label.height() <= 0:
            return
        self.label.setPixmap(
            self._pixmap.scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_display()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Object Detection")

        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.availableGeometry()
            width = int(geometry.width() * 0.85)
            height = int(geometry.height() * 0.85)
            self.resize(width, height)
            self.move(
                geometry.x() + (geometry.width() - width) // 2,
                geometry.y() + (geometry.height() - height) // 2,
            )
        else:
            self.resize(1400, 840)

        self.db_path = init_db()
        self.config = load_config()
        self.detector = Detector()
        self.roi_manager = ROIManager()
        self.rules: List[RuleDefinition] = []
        self.active_rule: RuleDefinition = ensure_rule_exists()
        self._selected_roi_ids: set[int] = set()
        self._selected_roi_shapes: List[ROIShape] = []
        self._all_roi_shapes: List[ROIShape] = []
        self._active_batch_id: Optional[int] = None
        self._active_model_id: Optional[int] = None
        self._recent_limit = 50
        self._recent_events: deque = deque(maxlen=self._recent_limit)
        self.analytics_result: Optional[AnalyticsResult] = None
        self.analytics_output_dir = Path("exports")
        self.current_review_inspection_id: Optional[int] = None
        self._active_model_info: Optional[ModelInfo] = None
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[VideoWorker] = None
        self.camera_manager: Optional[CameraManager] = None
        self.available_cameras = enumerate_cameras()
        self._last_fps = 0.0
        self._last_inference_ms = 0.0

        self._current_frame: Optional[np.ndarray] = None
        self._latest_pixmap: Optional[QPixmap] = None
        self.camera_popup: Optional[CameraPopupWindow] = None
        self._camera_popup_auto_opened = False

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
        review_widget = self._build_review_tab()
        analytics_widget = self._build_analytics_tab()
        registry_widget = self._build_registry_tab()
        settings_widget = self._build_settings_tab()
        self.tab_widget.addTab(detection_widget, "Detection")
        self.tab_widget.addTab(training_widget, "Training")
        self.tab_widget.addTab(review_widget, "Review & Relabel")
        self.tab_widget.addTab(analytics_widget, "Analytics & Reports")
        self.tab_widget.addTab(registry_widget, "Model Registry")
        self.tab_widget.addTab(settings_widget, "Settings")

        splitter.addWidget(self.tab_widget)
        splitter.addWidget(self._build_video_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        refresh_action = QAction("Refresh Cameras", self)
        refresh_action.triggered.connect(self.refresh_camera_items)
        self.menuBar().addAction(refresh_action)


    def _build_detection_tab(self) -> QWidget:
        control_widget = QWidget()
        outer_layout = QVBoxLayout(control_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)

        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)

        control_layout = QVBoxLayout(scroll_content)
        control_layout.setContentsMargins(12, 12, 12, 12)
        control_layout.setSpacing(12)

        source_group = QGroupBox("Camera Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)

        self.camera_combo = QComboBox()
        self.refresh_camera_items()

        source_form = QFormLayout()
        source_form.setLabelAlignment(Qt.AlignLeft)
        source_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        source_form.addRow("Camera Devices", self.camera_combo)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("http://<phone-ip>:8080/video")
        source_form.addRow("Stream URL", self.url_input)

        source_layout.addLayout(source_form)

        btn_row = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        btn_row.addWidget(self.connect_button)
        btn_row.addWidget(self.disconnect_button)
        btn_row.addStretch()
        source_layout.addLayout(btn_row)

        control_layout.addWidget(source_group)

        model_group = QGroupBox("Model & Classes")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)

        self.model_combo = QComboBox()
        for label, path_value in PRESET_MODELS:
            self.model_combo.addItem(label, path_value)
        self.model_combo.addItem("Model kustom (pilih file .pt)", CUSTOM_MODEL_SENTINEL)
        model_layout.addWidget(self.model_combo)

        self.model_picker = FilePicker("Select YOLO model", "PyTorch Model (*.pt)", DEFAULT_MODEL)
        model_layout.addWidget(self.model_picker)

        self.load_model_button = QPushButton("Load Model")
        model_layout.addWidget(self.load_model_button)

        self.confidence_slider = ConfidenceSlider()
        model_layout.addWidget(self.confidence_slider)

        self.class_selection = ClassSelection()
        self.class_selection.setEnabled(False)
        model_layout.addWidget(self.class_selection)

        control_layout.addWidget(model_group)

        detection_group = QGroupBox("Detection Controls")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setSpacing(8)

        detection_buttons = QHBoxLayout()
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.snapshot_button = QPushButton("Take Snapshot")
        detection_buttons.addWidget(self.start_button)
        detection_buttons.addWidget(self.stop_button)
        detection_buttons.addWidget(self.snapshot_button)
        detection_layout.addLayout(detection_buttons)

        self.record_checkbox = QCheckBox("Record annotated video")
        detection_layout.addWidget(self.record_checkbox)

        control_layout.addWidget(detection_group)

        override_group = QGroupBox("Manual Override & Notes")
        override_layout = QVBoxLayout(override_group)
        override_layout.setSpacing(8)

        override_row = QHBoxLayout()
        self.override_pass_button = QPushButton("Override PASS")
        self.override_fail_button = QPushButton("Override FAIL")
        self.clear_override_button = QPushButton("Clear Override")
        override_row.addWidget(self.override_pass_button)
        override_row.addWidget(self.override_fail_button)
        override_row.addWidget(self.clear_override_button)
        override_layout.addLayout(override_row)

        self.add_note_button = QPushButton("Add Note")
        override_layout.addWidget(self.add_note_button)

        control_layout.addWidget(override_group)

        batch_group = QGroupBox("Batch / Lot")
        batch_form = QFormLayout(batch_group)
        batch_form.setLabelAlignment(Qt.AlignLeft)
        batch_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.batch_lot_input = QLineEdit()
        self.batch_line_input = QLineEdit()
        self.batch_shift_input = QLineEdit()
        self.batch_operator_input = QLineEdit()
        self.batch_notes_input = QLineEdit()
        batch_form.addRow("Lot ID", self.batch_lot_input)
        batch_form.addRow("Line", self.batch_line_input)
        batch_form.addRow("Shift", self.batch_shift_input)
        batch_form.addRow("Operator", self.batch_operator_input)
        batch_form.addRow("Notes", self.batch_notes_input)
        control_layout.addWidget(batch_group)

        qc_group = QGroupBox("Quality Control")
        qc_layout = QVBoxLayout(qc_group)
        qc_layout.setSpacing(8)

        rule_row = QHBoxLayout()
        self.qc_rule_combo = QComboBox()
        self.rule_reload_button = QPushButton("Reload")
        rule_row.addWidget(self.qc_rule_combo)
        rule_row.addWidget(self.rule_reload_button)
        qc_layout.addLayout(rule_row)

        roi_header = QHBoxLayout()
        roi_header.addWidget(QLabel("ROI Presets"))
        roi_header.addStretch()
        self.roi_refresh_button = QPushButton("Refresh")
        roi_header.addWidget(self.roi_refresh_button)
        qc_layout.addLayout(roi_header)

        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(120)
        qc_layout.addWidget(self.roi_list)

        control_layout.addWidget(qc_group)

        metrics_group = QGroupBox("Inspection Status")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.setSpacing(6)
        self.batch_status_label = QLabel("Batch: -")
        self.status_label = QLabel("Status: -")
        self.pass_count_label = QLabel("Pass: 0")
        self.fail_count_label = QLabel("Fail: 0")
        counters_row = QHBoxLayout()
        counters_row.addWidget(self.pass_count_label)
        counters_row.addWidget(self.fail_count_label)
        counters_row.addStretch()
        metrics_layout.addWidget(self.batch_status_label)
        metrics_layout.addWidget(self.status_label)
        metrics_layout.addLayout(counters_row)
        metrics_layout.addWidget(QLabel("Defects (current batch)"))
        self.defect_list = QListWidget()
        self.defect_list.setMaximumHeight(100)
        metrics_layout.addWidget(self.defect_list)
        control_layout.addWidget(metrics_group)

        detections_group = QGroupBox("Recent Detections")
        detections_layout = QVBoxLayout(detections_group)
        self.detections_table = QTableWidget(0, 4)
        self.detections_table.setHorizontalHeaderLabels(["Time", "Class", "Conf", "Status"])
        header = self.detections_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.detections_table.verticalHeader().setVisible(False)
        self.detections_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.detections_table.setMaximumHeight(200)
        detections_layout.addWidget(self.detections_table)
        control_layout.addWidget(detections_group)

        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_view)
        control_layout.addWidget(log_group)

        control_layout.addStretch()

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

        extras_row = QHBoxLayout()
        self.training_open_results_button = QPushButton("Open Results Folder")
        self.training_register_button = QPushButton("Register Trained Model")
        self.training_create_colab_button = QPushButton("Create Colab Notebook")
        extras_row.addWidget(self.training_open_results_button)
        extras_row.addWidget(self.training_register_button)
        extras_row.addWidget(self.training_create_colab_button)
        extras_row.addStretch()
        layout.addLayout(extras_row)

        self.training_load_model_button = QPushButton("Load Trained Model")
        self.training_load_model_button.setEnabled(False)
        layout.addWidget(self.training_load_model_button)

        return training_widget


    def _build_review_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        left_panel = QVBoxLayout()
        filter_row = QHBoxLayout()
        self.review_refresh_button = QPushButton("Refresh")
        self.review_fail_only_checkbox = QCheckBox("Fail only")
        self.review_fail_only_checkbox.setChecked(True)
        filter_row.addWidget(self.review_refresh_button)
        filter_row.addWidget(self.review_fail_only_checkbox)
        filter_row.addStretch()
        left_panel.addLayout(filter_row)

        self.review_list = QListWidget()
        left_panel.addWidget(self.review_list, 1)

        review_buttons = QHBoxLayout()
        self.review_save_button = QPushButton("Save Truth")
        self.review_export_button = QPushButton("Export to Dataset")
        review_buttons.addWidget(self.review_save_button)
        review_buttons.addWidget(self.review_export_button)
        review_buttons.addStretch()
        left_panel.addLayout(review_buttons)

        layout.addLayout(left_panel, 1)

        self.review_view = ReviewWidget()
        layout.addWidget(self.review_view, 2)

        return widget

    def _build_analytics_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        filters = QHBoxLayout()
        self.analytics_start_date = QDateEdit()
        self.analytics_start_date.setCalendarPopup(True)
        self.analytics_start_date.setDate(QDate.currentDate().addDays(-7))
        self.analytics_end_date = QDateEdit()
        self.analytics_end_date.setCalendarPopup(True)
        self.analytics_end_date.setDate(QDate.currentDate())
        filters.addWidget(QLabel("Start"))
        filters.addWidget(self.analytics_start_date)
        filters.addWidget(QLabel("End"))
        filters.addWidget(self.analytics_end_date)
        self.analytics_lot_input = QLineEdit()
        self.analytics_lot_input.setPlaceholderText("Lot ID")
        filters.addWidget(self.analytics_lot_input)
        self.analytics_line_input = QLineEdit()
        self.analytics_line_input.setPlaceholderText("Line")
        filters.addWidget(self.analytics_line_input)
        self.analytics_shift_input = QLineEdit()
        self.analytics_shift_input.setPlaceholderText("Shift")
        filters.addWidget(self.analytics_shift_input)
        filters.addStretch()
        layout.addLayout(filters)

        analytics_buttons = QHBoxLayout()
        self.analytics_run_button = QPushButton("Run Analytics")
        self.analytics_export_csv_button = QPushButton("Export CSV")
        self.analytics_export_pdf_button = QPushButton("Export PDF")
        analytics_buttons.addWidget(self.analytics_run_button)
        analytics_buttons.addWidget(self.analytics_export_csv_button)
        analytics_buttons.addWidget(self.analytics_export_pdf_button)
        analytics_buttons.addStretch()
        layout.addLayout(analytics_buttons)

        self.analytics_summary = QTextEdit()
        self.analytics_summary.setReadOnly(True)
        self.analytics_summary.setPlaceholderText("Analytics results will appear here...")
        layout.addWidget(self.analytics_summary)

        charts_layout = QHBoxLayout()
        self.analytics_chart_labels: Dict[str, QLabel] = {}
        for key in ("defects_by_class", "defect_trend", "pass_fail"):
            label = QLabel("Chart will appear here")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(200, 150)
            label.setStyleSheet("border: 1px solid #444;")
            charts_layout.addWidget(label)
            self.analytics_chart_labels[key] = label
        layout.addLayout(charts_layout)

        return widget

    def _build_registry_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        registry_buttons = QHBoxLayout()
        self.registry_refresh_button = QPushButton("Refresh")
        self.registry_import_button = QPushButton("Import Model")
        self.registry_set_active_button = QPushButton("Set Active")
        self.registry_delete_button = QPushButton("Delete")
        registry_buttons.addWidget(self.registry_refresh_button)
        registry_buttons.addWidget(self.registry_import_button)
        registry_buttons.addWidget(self.registry_set_active_button)
        registry_buttons.addWidget(self.registry_delete_button)
        registry_buttons.addStretch()
        layout.addLayout(registry_buttons)

        self.registry_table = QTableWidget(0, 6)
        self.registry_table.setHorizontalHeaderLabels(["ID", "Name", "Type", "Path", "Hash", "Active"])
        header = self.registry_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.registry_table.verticalHeader().setVisible(False)
        self.registry_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.registry_table)

        return widget

    def _build_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        rule_group = QGroupBox("QC Rule JSON")
        rule_layout = QVBoxLayout(rule_group)
        self.rule_editor = QTextEdit()
        self.rule_editor.setPlaceholderText("Edit QC rule JSON here")
        rule_layout.addWidget(self.rule_editor)
        rule_buttons = QHBoxLayout()
        self.rule_save_button = QPushButton("Save Rule")
        self.rule_new_button = QPushButton("New Rule")
        rule_buttons.addWidget(self.rule_save_button)
        rule_buttons.addWidget(self.rule_new_button)
        rule_buttons.addStretch()
        rule_layout.addLayout(rule_buttons)
        layout.addWidget(rule_group)

        roi_group = QGroupBox("ROI Presets")
        roi_layout = QVBoxLayout(roi_group)
        self.settings_roi_list = QListWidget()
        roi_layout.addWidget(self.settings_roi_list)
        roi_buttons = QHBoxLayout()
        self.settings_add_roi_button = QPushButton("Add ROI")
        self.settings_delete_roi_button = QPushButton("Delete ROI")
        roi_buttons.addWidget(self.settings_add_roi_button)
        roi_buttons.addWidget(self.settings_delete_roi_button)
        roi_buttons.addStretch()
        roi_layout.addLayout(roi_buttons)
        layout.addWidget(roi_group)

        layout.addStretch()
        return widget


    def _build_video_panel(self) -> QWidget:
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.setSpacing(12)

        self.video_label = QLabel("Camera feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #202020; color: #ffffff;")
        self.video_label.setMinimumSize(640, 360)
        video_layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()
        controls.addStretch()
        self.open_camera_popup_button = QPushButton("Open Camera Popup")
        controls.addWidget(self.open_camera_popup_button)
        controls.addStretch()
        video_layout.addLayout(controls)

        return video_widget

    def on_open_camera_popup(self) -> None:
        if self.camera_popup and self.camera_popup.isVisible():
            self.camera_popup.close()
        else:
            self._ensure_camera_popup()

    def _ensure_camera_popup(self) -> None:
        if self.camera_popup and self.camera_popup.isVisible():
            return
        if not self.camera_popup:
            self.camera_popup = CameraPopupWindow(self)
            self.camera_popup.closed.connect(self._on_camera_popup_closed)
        self.camera_popup.show()
        self.camera_popup.raise_()
        self.camera_popup.activateWindow()
        if self._latest_pixmap:
            self.camera_popup.set_pixmap(self._latest_pixmap)
        self.open_camera_popup_button.setText("Close Camera Popup")

    def _on_camera_popup_closed(self) -> None:
        self.open_camera_popup_button.setText("Open Camera Popup")
        self.camera_popup = None

    def _refresh_video_outputs(self) -> None:
        if not self._latest_pixmap:
            return
        if self.video_label.width() <= 0 or self.video_label.height() <= 0:
            return
        self.video_label.setPixmap(
            self._latest_pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        if self.camera_popup and self.camera_popup.isVisible():
            self.camera_popup.set_pixmap(self._latest_pixmap)

    def _clear_video_outputs(self) -> None:
        self._current_frame = None
        self._latest_pixmap = None
        self._camera_popup_auto_opened = False
        self.video_label.clear()
        self.video_label.setText("Camera feed will appear here")
        if self.camera_popup and self.camera_popup.isVisible():
            self.camera_popup.clear()

    def _connect_signals(self) -> None:
        self.connect_button.clicked.connect(self.on_connect)
        self.disconnect_button.clicked.connect(self.on_disconnect)
        self.start_button.clicked.connect(self.on_start_detection)
        self.stop_button.clicked.connect(self.on_stop_detection)
        self.snapshot_button.clicked.connect(self.on_snapshot)
        self.open_camera_popup_button.clicked.connect(self.on_open_camera_popup)
        self.load_model_button.clicked.connect(self.on_load_model)
        self.model_combo.currentIndexChanged.connect(self.on_model_choice_changed)
        self.model_picker.valueChanged.connect(self.on_model_path_changed)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.class_selection.valueChanged.connect(self.on_class_selection_changed)
        self.record_checkbox.stateChanged.connect(self.on_record_toggle)
        self.override_pass_button.clicked.connect(lambda: self.on_override("PASS"))
        self.override_fail_button.clicked.connect(lambda: self.on_override("FAIL"))
        self.clear_override_button.clicked.connect(lambda: self.on_override(None))
        self.add_note_button.clicked.connect(self.on_add_note)
        self.rule_reload_button.clicked.connect(self.load_rules_to_combo)
        self.qc_rule_combo.currentIndexChanged.connect(self.on_rule_changed)
        self.roi_refresh_button.clicked.connect(self.reload_roi_presets)
        self.roi_list.itemChanged.connect(self.on_roi_selection_changed)
        self.detections_table.itemDoubleClicked.connect(self.on_detection_item_double_clicked)
        self.batch_line_input.textChanged.connect(self.on_batch_line_changed)
        self.review_refresh_button.clicked.connect(self.refresh_review_list)
        self.review_list.currentItemChanged.connect(self.on_review_item_changed)
        self.review_view.saveRequested.connect(self.on_review_save_truth)
        self.review_view.exportRequested.connect(self.on_review_export)
        self.review_save_button.clicked.connect(self.on_review_save_truth)
        self.review_export_button.clicked.connect(self.on_review_export)
        self.review_fail_only_checkbox.stateChanged.connect(lambda _state: self.refresh_review_list())
        self.analytics_run_button.clicked.connect(self.on_run_analytics)
        self.analytics_export_csv_button.clicked.connect(self.on_export_analytics_csv)
        self.analytics_export_pdf_button.clicked.connect(self.on_export_analytics_pdf)
        self.registry_refresh_button.clicked.connect(self.refresh_registry_table)
        self.registry_import_button.clicked.connect(self.on_registry_import)
        self.registry_set_active_button.clicked.connect(self.on_registry_set_active)
        self.registry_delete_button.clicked.connect(self.on_registry_delete)
        self.registry_table.itemSelectionChanged.connect(self.on_registry_selection_changed)
        self.rule_save_button.clicked.connect(self.on_save_rule_clicked)
        self.rule_new_button.clicked.connect(self.on_new_rule_clicked)
        self.settings_add_roi_button.clicked.connect(self.on_settings_add_roi)
        self.settings_delete_roi_button.clicked.connect(self.on_settings_delete_roi)
        self.settings_roi_list.itemSelectionChanged.connect(self.on_settings_roi_selected)
        self.training_start_button.clicked.connect(self.on_start_training)
        self.training_stop_button.clicked.connect(self.on_stop_training)
        self.training_load_model_button.clicked.connect(self.on_load_trained_model)
        self.training_open_results_button.clicked.connect(self.on_open_training_results)
        self.training_register_button.clicked.connect(self.on_register_trained_model)
        self.training_create_colab_button.clicked.connect(self.on_create_colab_notebook)
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

        self.load_rules_to_combo()
        self.reload_roi_presets()
        self.status_label.setText("Status: -")
        self.batch_status_label.setText("Batch: -")
        self.pass_count_label.setText("Pass: 0")
        self.fail_count_label.setText("Fail: 0")
        self.defect_list.clear()
        self.update_rule_editor()
        self.refresh_settings_roi_list()
        self.refresh_registry_table()
        self.refresh_review_list()

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
        self.worker.inspectionRecorded.connect(self.on_inspection_recorded)
        self.worker.countersUpdated.connect(self.on_counters_updated)
        self.worker.batchChanged.connect(self.on_batch_changed)
        if self.active_rule:
            self.worker.update_rule(self.active_rule)
        if self._selected_roi_shapes:
            self.worker.update_roi(self._selected_roi_shapes)
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
        self._clear_video_outputs()

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
        batch_context = self._build_batch_context()
        if batch_context is None and self._has_partial_batch_metadata():
            self.status_bar.showMessage("Lot ID and Line are required to log batch data.", 5000)
        roi_shapes = list(self._selected_roi_shapes)
        rule = self.active_rule
        self.worker.enable_detection(
            confidence,
            class_ids,
            record,
            batch=batch_context,
            rule=rule,
            roi=roi_shapes,
            model_id=self._active_model_id,
        )
        if batch_context:
            self.batch_status_label.setText(
                f"Batch pending | Lot {batch_context.lot_id} | Line {batch_context.line}"
            )
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
        self.status_label.setText("Status: stopped")
        self.status_label.setStyleSheet("")
        self.status_bar.showMessage("Detection stopped")

    def _build_batch_context(self) -> Optional[BatchContext]:
        lot_id = self.batch_lot_input.text().strip()
        line = self.batch_line_input.text().strip()
        shift = self.batch_shift_input.text().strip()
        operator = self.batch_operator_input.text().strip()
        notes = self.batch_notes_input.text().strip()
        if not lot_id and not line and not shift and not operator and not notes:
            return None
        if not lot_id or not line:
            return None
        return BatchContext(
            lot_id=lot_id,
            line=line,
            shift=shift or "",
            operator=operator or "",
            notes=notes,
        )

    def _has_partial_batch_metadata(self) -> bool:
        return any(
            field.strip()
            for field in (
                self.batch_lot_input.text(),
                self.batch_line_input.text(),
                self.batch_shift_input.text(),
                self.batch_operator_input.text(),
                self.batch_notes_input.text(),
            )
        )

    def _push_batch_update(self) -> None:
        if self.worker:
            self.worker.update_batch(self._build_batch_context())

    def load_rules_to_combo(self) -> None:
        if not hasattr(self, "qc_rule_combo"):
            return
        with session_scope() as session:
            rules = list_rules(session)
        if not rules:
            self.active_rule = ensure_rule_exists()
            with session_scope() as session:
                rules = list_rules(session)
        self.rules = rules
        active_id = self.active_rule.id if self.active_rule else None
        self.qc_rule_combo.blockSignals(True)
        self.qc_rule_combo.clear()
        active_index = 0
        for index, rule in enumerate(rules):
            label = f"{rule.name} (min {rule.min_conf:.2f})"
            self.qc_rule_combo.addItem(label, rule.id)
            if active_id == rule.id:
                active_index = index
        if rules:
            self.qc_rule_combo.setCurrentIndex(active_index)
            self.active_rule = rules[active_index]
        self.qc_rule_combo.blockSignals(False)
        if self.worker and self.active_rule:
            self.worker.update_rule(self.active_rule)
        self.update_rule_editor()

    def on_rule_changed(self, index: int) -> None:
        if not self.rules:
            return
        if index < 0 or index >= self.qc_rule_combo.count():
            return
        rule_id = self.qc_rule_combo.itemData(index)
        if rule_id is None:
            return
        try:
            rule_id_int = int(rule_id)
        except (TypeError, ValueError):
            return
        with session_scope() as session:
            rule = set_active_rule(session, rule_id_int)
        if not rule:
            QMessageBox.warning(self, "Rule missing", "Selected QC rule could not be loaded.")
            return
        self.active_rule = rule
        self.rules = [rule if existing.id == rule.id else existing for existing in self.rules]
        if self.worker:
            self.worker.update_rule(rule)
        self.update_rule_editor()
        self.status_bar.showMessage(f"Active rule set to {rule.name}", 3000)

    def reload_roi_presets(self) -> None:
        if not hasattr(self, "roi_list"):
            return
        line_filter = self.batch_line_input.text().strip() if hasattr(self, "batch_line_input") else ""
        shapes = self.roi_manager.list_presets(line_filter or None)
        self._all_roi_shapes = shapes
        valid_ids = {shape.id for shape in shapes if shape.id is not None}
        self._selected_roi_ids = {roi_id for roi_id in self._selected_roi_ids if roi_id in valid_ids}
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for shape in shapes:
            item = QListWidgetItem(f"{shape.line} - {shape.name} ({shape.kind})")
            item.setData(Qt.UserRole, shape)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            if shape.id in self._selected_roi_ids:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.roi_list.addItem(item)
        self.roi_list.blockSignals(False)
        self._selected_roi_shapes = [shape for shape in shapes if shape.id in self._selected_roi_ids]
        if self.worker:
            self.worker.update_roi(self._selected_roi_shapes)

    def on_roi_selection_changed(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        shape = item.data(Qt.UserRole)
        if not isinstance(shape, ROIShape):
            return
        if shape.id is not None:
            if item.checkState() == Qt.Checked:
                self._selected_roi_ids.add(shape.id)
            else:
                self._selected_roi_ids.discard(shape.id)
        self._selected_roi_shapes = [s for s in self._all_roi_shapes if s.id in self._selected_roi_ids]
        if self.worker:
            self.worker.update_roi(self._selected_roi_shapes)
        self.status_bar.showMessage(f"ROI selected: {len(self._selected_roi_shapes)} active", 2000)

    def on_batch_line_changed(self, _text: str) -> None:
        self.reload_roi_presets()
        self._push_batch_update()

    def on_add_note(self) -> None:
        if not self.worker:
            QMessageBox.information(self, "Detection inactive", "Connect to a camera and start detection before adding notes.")
            return
        text, ok = QInputDialog.getMultiLineText(self, "Add Inspection Note", "Note for next inspection:")
        if not ok:
            return
        note = text.strip()
        if not note:
            return
        self.worker.queue_note(note)
        self.append_log(f"Note queued: {note}")
        self.status_bar.showMessage("Note will tag the next inspection", 4000)

    def on_override(self, status: Optional[str]) -> None:
        if not self.worker:
            QMessageBox.information(self, "Detection inactive", "Connect to a camera before applying overrides.")
            return
        try:
            self.worker.override_next(status)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid override", str(exc))
            return
        if status:
            message = f"Next inspection forced to {status.upper()}"
        else:
            message = "Pending override cleared"
        self.append_log(message)
        self.status_bar.showMessage(message, 4000)

    def on_detection_item_double_clicked(self, item: QTableWidgetItem) -> None:
        if item is None:
            return
        row = item.row()
        event_item = self.detections_table.item(row, 0)
        if event_item is None:
            return
        event = event_item.data(Qt.UserRole) or {}
        frame_path = event.get("frame_path")
        if not frame_path:
            QMessageBox.information(self, "No snapshot", "This inspection does not have an associated snapshot.")
            return
        path = Path(str(frame_path))
        if not path.exists():
            QMessageBox.warning(self, "File missing", f"Snapshot not found: {path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def on_inspection_recorded(self, event: Dict[str, object]) -> None:
        status = str(event.get("pass_fail", "")) or "-"
        self._set_status_label(status)
        batch_id = event.get("batch_id")
        try:
            self._active_batch_id = int(batch_id) if batch_id is not None else self._active_batch_id
        except (TypeError, ValueError):
            self._active_batch_id = None
        if status.upper() == "FAIL":
            QApplication.beep()
            self.status_bar.showMessage("Inspection marked as FAIL", 5000)
        else:
            self.status_bar.showMessage(f"Inspection marked as {status}", 3000)
        note = event.get("note")
        if note:
            self.append_log(f"Inspection note: {note}")
        frame_path = event.get("frame_path")
        if frame_path:
            self.append_log(f"Snapshot saved to {frame_path}")

    def on_counters_updated(self, payload: Dict[str, object]) -> None:
        try:
            pass_count = int(payload.get("pass_count") or 0)
        except (TypeError, ValueError):
            pass_count = 0
        try:
            fail_count = int(payload.get("fail_count") or 0)
        except (TypeError, ValueError):
            fail_count = 0
        self.pass_count_label.setText(f"Pass: {pass_count}")
        self.fail_count_label.setText(f"Fail: {fail_count}")
        defects = payload.get("defect_counts") or {}
        self.defect_list.clear()
        if isinstance(defects, dict):
            for label, count in sorted(defects.items(), key=lambda item: (-int(item[1] or 0) if item[1] is not None else 0, str(item[0]))):
                try:
                    count_value = int(count)
                except (TypeError, ValueError):
                    count_value = 0
                self.defect_list.addItem(f"{label}: {count_value}")
        recent = payload.get("recent")
        if isinstance(recent, list):
            self._recent_events.clear()
            for event in recent:
                self._recent_events.append(event)
            self._refresh_recent_detections_table()
        batch_id = payload.get("batch_id")
        try:
            self._active_batch_id = int(batch_id) if batch_id is not None else self._active_batch_id
        except (TypeError, ValueError):
            self._active_batch_id = None

    def on_batch_changed(self, payload: Dict[str, object]) -> None:
        status = payload.get("status")
        if status == "opened":
            batch_id = payload.get("batch_id")
            lot = payload.get("lot_id") or "-"
            line = payload.get("line") or "-"
            self.batch_status_label.setText(f"Batch {batch_id} | Lot {lot} | Line {line}")
            try:
                self._active_batch_id = int(batch_id) if batch_id is not None else None
            except (TypeError, ValueError):
                self._active_batch_id = None
            self.status_bar.showMessage("Batch opened", 3000)
        elif status == "closed":
            self.batch_status_label.setText("Batch closed")
            self._active_batch_id = None
            self.status_bar.showMessage("Batch closed", 3000)

    def _refresh_recent_detections_table(self) -> None:
        self.detections_table.setRowCount(0)
        for event in list(self._recent_events):
            row = self.detections_table.rowCount()
            self.detections_table.insertRow(row)
            time_item = QTableWidgetItem(self._format_event_time(event.get("created_at")))
            time_item.setData(Qt.UserRole, event)
            class_text, conf_text = self._extract_detection_summary(event.get("detections") or [])
            self.detections_table.setItem(row, 0, time_item)
            self.detections_table.setItem(row, 1, QTableWidgetItem(class_text))
            self.detections_table.setItem(row, 2, QTableWidgetItem(conf_text))
            self.detections_table.setItem(row, 3, QTableWidgetItem(str(event.get("pass_fail", ""))))

    def _format_event_time(self, value: object) -> str:
        if not value:
            return "-"
        if isinstance(value, datetime):
            return value.strftime("%H:%M:%S")
        if isinstance(value, str):
            cleaned = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(cleaned)
            except ValueError:
                return value
            return parsed.strftime("%H:%M:%S")
        return str(value)

    def _extract_detection_summary(self, detections: Sequence[Dict[str, object]]) -> Tuple[str, str]:
        if not detections:
            return "-", "-"
        first = detections[0]
        label = str(first.get("label", "")) or "-"
        confidence = first.get("confidence")
        try:
            conf_value = float(confidence)
            conf_text = f"{conf_value:.2f}"
        except (TypeError, ValueError):
            conf_text = "-"
        return label, conf_text

    def _set_status_label(self, status: str) -> None:
        normalized = status.upper() if isinstance(status, str) else ""
        if normalized == "FAIL":
            self.status_label.setText("Status: FAIL")
            self.status_label.setStyleSheet("color: #d62728; font-weight: bold;")
        elif normalized == "PASS":
            self.status_label.setText("Status: PASS")
            self.status_label.setStyleSheet("color: #2ca02c; font-weight: bold;")
        else:
            self.status_label.setText(f"Status: {status or '-'}")
            self.status_label.setStyleSheet("")
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
        self._latest_pixmap = QPixmap.fromImage(image)
        self._refresh_video_outputs()
        if not self._camera_popup_auto_opened:
            self._ensure_camera_popup()
            self._camera_popup_auto_opened = True

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_video_outputs()



    def on_stats_updated(self, fps: float, inference_ms: float) -> None:
        self._last_fps = fps
        self._last_inference_ms = inference_ms
        self.status_bar.showMessage(f"FPS: {fps:.1f} | Inference: {inference_ms:.1f} ms")

    def on_worker_error(self, message: str) -> None:
        self.append_log(f"Error: {message}")
        self.status_bar.showMessage(message)
        self._clear_video_outputs()

    def on_worker_finished(self) -> None:
        self.append_log("Worker finished")
        self._clear_video_outputs()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.worker = None
        self.worker_thread = None
        self.camera_manager = None
        self.status_label.setText("Status: -")
        self.status_label.setStyleSheet("")
        self.batch_status_label.setText("Batch: -")

    def on_open_training_results(self) -> None:
        path = self.training_best_weights
        if not path or not Path(path).exists():
            last = self.config.get("last_trained_weights")
            path = Path(last) if last else None
        target = Path(path).parent if path else Path("runs")
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
        else:
            QMessageBox.information(self, "Not found", "No training results available yet.")

    def on_register_trained_model(self) -> None:
        path = self.training_best_weights
        if not path or not Path(path).exists():
            last = self.config.get("last_trained_weights")
            if not last:
                QMessageBox.warning(self, "Not found", "No trained weights to register yet.")
                return
            path = Path(last)
        try:
            info = register_model_path(Path(path), set_active=False)
        except Exception as exc:
            LOGGER.exception("Register model failed")
            QMessageBox.critical(self, "Registration failed", str(exc))
            return
        self.append_log(f"Registered trained model: {info.name}")
        self.refresh_registry_table()

    def on_create_colab_notebook(self) -> None:
        dataset_path = Path(self.training_data_picker.text()).resolve()
        if not dataset_path.exists():
            QMessageBox.warning(self, "Dataset not found", f"Dataset file not found: {dataset_path}")
            return
        output_dir = Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)
        config = ColabConfig(
            dataset_path=dataset_path,
            output_dir=output_dir,
            base_model=self.training_model_picker.text() or DEFAULT_MODEL,
            epochs=int(self.training_epochs_spin.value()),
            imgsz=int(self.training_imgsz_spin.value()),
            batch=int(self.training_batch_spin.value()),
        )
        notebook_path = output_dir / f"qc_training_{timestamp_for_file()}.ipynb"
        try:
            generate_notebook(config, notebook_path)
        except Exception as exc:
            LOGGER.exception("Failed to generate Colab notebook")
            QMessageBox.critical(self, "Generation failed", str(exc))
            return
        self.status_bar.showMessage(f"Colab notebook created: {notebook_path}", 5000)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(notebook_path.resolve())))
    def refresh_review_list(self) -> None:
        with session_scope() as session:
            query = session.query(Inspection).order_by(Inspection.created_at.desc())
            if isinstance(getattr(self, 'review_fail_only_checkbox', None), QCheckBox) and self.review_fail_only_checkbox.isChecked():
                query = query.filter(Inspection.pass_fail == "FAIL")
            inspections = query.limit(200).all()
        self.review_list.blockSignals(True)
        self.review_list.clear()
        for inspection in inspections:
            created = inspection.created_at.strftime("%Y-%m-%d %H:%M:%S") if inspection.created_at else "-"
            item = QListWidgetItem(f"{created} | {inspection.pass_fail}")
            item.setData(Qt.UserRole, inspection.id)
            item.setData(Qt.UserRole + 1, inspection.frame_path)
            item.setData(Qt.UserRole + 2, inspection.batch_id)
            self.review_list.addItem(item)
        self.review_list.blockSignals(False)
        if inspections:
            self.review_list.setCurrentRow(0)

    def on_review_item_changed(self, current: QListWidgetItem, _previous: Optional[QListWidgetItem] = None) -> None:
        if current is None:
            self.current_review_inspection_id = None
            return
        inspection_id = current.data(Qt.UserRole)
        if inspection_id is None:
            return
        self.current_review_inspection_id = int(inspection_id)
        frame_path_data = current.data(Qt.UserRole + 1)
        if frame_path_data:
            frame_path = Path(str(frame_path_data))
            if frame_path.exists():
                self.review_view.load_from_path(frame_path)
        with session_scope() as session:
            detections = session.query(Detection).filter(Detection.inspection_id == self.current_review_inspection_id).all()
        boxes = []
        for det in detections:
            rect = QRectF(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1)
            boxes.append(DetectionBox(cls=det.cls, conf=det.conf, rect=rect))
        self.review_view.set_detections(boxes)

    def on_review_save_truth(self) -> None:
        if self.current_review_inspection_id is None:
            QMessageBox.information(self, "No selection", "Select an inspection first.")
            return
        boxes = self.review_view.to_label_boxes()
        with session_scope() as session:
            session.query(ReviewTruth).filter(ReviewTruth.inspection_id == self.current_review_inspection_id).delete()
            for box in boxes:
                session.add(
                    ReviewTruth(
                        inspection_id=self.current_review_inspection_id,
                        cls=box.cls,
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        status="edited",
                    )
                )
        self.status_bar.showMessage("Review truth saved", 3000)
        self.append_log(f"Saved review truth for inspection {self.current_review_inspection_id}")

    def on_review_export(self) -> None:
        if not self.review_list.count():
            QMessageBox.information(self, "No items", "No review items available for export.")
            return
        export_dir = self.analytics_output_dir / "dataset_reviewed"
        export_dir.mkdir(parents=True, exist_ok=True)
        selected_items = self.review_list.selectedItems() or [self.review_list.item(i) for i in range(self.review_list.count())]
        reviewed_items: List[ReviewedItem] = []
        with session_scope() as session:
            for item in selected_items:
                inspection_id = item.data(Qt.UserRole)
                frame_path_data = item.data(Qt.UserRole + 1)
                if inspection_id is None or not frame_path_data:
                    continue
                frame_path = Path(str(frame_path_data))
                if not frame_path.exists():
                    continue
                truths = session.query(ReviewTruth).filter(ReviewTruth.inspection_id == inspection_id).all()
                if truths:
                    boxes = [LabelBox(cls=t.cls, x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2) for t in truths]
                else:
                    detections = session.query(Detection).filter(Detection.inspection_id == inspection_id).all()
                    boxes = [LabelBox(cls=d.cls, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2) for d in detections]
                reviewed_items.append(ReviewedItem(inspection_id=inspection_id, image_path=frame_path, boxes=boxes))
        if not reviewed_items:
            QMessageBox.warning(self, "No data", "No review data available for export.")
            return
        classes = list(self.detector.names.values()) if self.detector.names else []
        export_reviewed_items(reviewed_items, export_dir, classes)
        self.status_bar.showMessage(f"Exported {len(reviewed_items)} items to {export_dir}", 5000)

    def on_run_analytics(self) -> None:
        start_qdate = self.analytics_start_date.date()
        end_qdate = self.analytics_end_date.date()
        if hasattr(start_qdate, "toPython"):
            start_date = start_qdate.toPython()
            end_date = end_qdate.toPython()
        else:
            start_date = datetime(start_qdate.year(), start_qdate.month(), start_qdate.day()).date()
            end_date = datetime(end_qdate.year(), end_qdate.month(), end_qdate.day()).date()
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())
        filters = AnalyticsFilters(
            start=start_dt,
            end=end_dt,
            lot_id=self.analytics_lot_input.text().strip() or None,
            line=self.analytics_line_input.text().strip() or None,
            shift=self.analytics_shift_input.text().strip() or None,
        )
        charts_dir = self.analytics_output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_analytics(filters, charts_dir)
        except Exception as exc:
            LOGGER.exception("Analytics failed")
            QMessageBox.critical(self, "Analytics failed", str(exc))
            return
        self.analytics_result = result
        self.display_analytics_result(result)
        self.status_bar.showMessage("Analytics completed", 3000)

    def display_analytics_result(self, result: AnalyticsResult) -> None:
        summary_lines = [
            f"Total inspections: {result.kpis.get('total_inspections', 0)}",
            f"Pass count: {result.kpis.get('pass_count', 0)}",
            f"Fail count: {result.kpis.get('fail_count', 0)}",
            f"Pass rate: {result.kpis.get('pass_rate', 0):.2f}%",
        ]
        top_defects = result.kpis.get("top_defects") or []
        if top_defects:
            summary_lines.append("Top defects:")
            for label, count in top_defects:
                summary_lines.append(f"  - {label}: {count}")
        self.analytics_summary.setPlainText("\n".join(summary_lines))
        for key, label in self.analytics_chart_labels.items():
            path = result.chart_paths.get(key)
            if path and Path(path).exists():
                pixmap = QPixmap(str(path))
                label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setToolTip(str(path))
                label.setText("")
            else:
                label.setPixmap(QPixmap())
                label.setText("No chart")

    def on_export_analytics_csv(self) -> None:
        if not self.analytics_result:
            QMessageBox.information(self, "No analytics", "Run analytics before exporting.")
            return
        export_dir = self.analytics_output_dir / "csv"
        export_dir.mkdir(parents=True, exist_ok=True)
        paths = export_csv(self.analytics_result, export_dir)
        self.status_bar.showMessage(f"CSV exported to {paths.detail_csv}", 5000)

    def on_export_analytics_pdf(self) -> None:
        if not self.analytics_result:
            QMessageBox.information(self, "No analytics", "Run analytics before exporting.")
            return
        pdf_dir = self.analytics_output_dir / "reports"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        metadata = ReportMetadata(
            title="QC Analytics Report",
            line=self.analytics_line_input.text().strip(),
            lot_id=self.analytics_lot_input.text().strip(),
            shift=self.analytics_shift_input.text().strip(),
            model_name=self.detector.model_path,
            rule_version=self.active_rule.version if self.active_rule else "",
        )
        pdf_path = pdf_dir / f"report_{timestamp_for_file()}.pdf"
        export_pdf(self.analytics_result, metadata, pdf_path)
        self.status_bar.showMessage(f"PDF exported to {pdf_path}", 5000)

    def refresh_registry_table(self) -> None:
        infos = list_model_info()
        self.registry_table.setRowCount(0)
        self._active_model_id = None
        self._active_model_info = None
        for info in infos:
            row = self.registry_table.rowCount()
            self.registry_table.insertRow(row)
            id_item = QTableWidgetItem(str(info.id))
            id_item.setData(Qt.UserRole, info)
            self.registry_table.setItem(row, 0, id_item)
            self.registry_table.setItem(row, 1, QTableWidgetItem(info.name))
            self.registry_table.setItem(row, 2, QTableWidgetItem(info.type))
            path_item = QTableWidgetItem(str(info.path))
            path_item.setToolTip(str(info.path))
            self.registry_table.setItem(row, 3, path_item)
            self.registry_table.setItem(row, 4, QTableWidgetItem(info.hash[:8]))
            active_item = QTableWidgetItem("Yes" if info.is_active else "")
            self.registry_table.setItem(row, 5, active_item)
            if info.is_active:
                self._active_model_id = info.id
                self._active_model_info = info
                self.registry_table.selectRow(row)

    def on_registry_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select model", str(Path.cwd()), "PyTorch Model (*.pt)")
        if not path:
            return
        try:
            info = register_model_path(Path(path), set_active=False)
        except Exception as exc:
            LOGGER.exception("Model import failed")
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self.append_log(f"Registered model: {info.name} -> {info.path}")
        self.refresh_registry_table()

    def _selected_registry_info(self) -> Optional[ModelInfo]:
        items = self.registry_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        info_item = self.registry_table.item(row, 0)
        if not info_item:
            return None
        info = info_item.data(Qt.UserRole)
        return info if isinstance(info, ModelInfo) else None

    def on_registry_set_active(self) -> None:
        info = self._selected_registry_info()
        if not info:
            QMessageBox.information(self, "Select model", "Select a model to set active.")
            return
        set_active(info.id)
        self._active_model_id = info.id
        self._active_model_info = info
        self.refresh_registry_table()
        self.status_bar.showMessage(f"Active model set to {info.name}", 3000)

    def on_registry_delete(self) -> None:
        info = self._selected_registry_info()
        if not info:
            QMessageBox.information(self, "Select model", "Select a model to delete.")
            return
        answer = QMessageBox.question(
            self,
            "Delete model",
            f"Delete model {info.name}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        with session_scope() as session:
            model = session.get(ModelRecord, info.id)
            if model:
                session.delete(model)
        self.refresh_registry_table()
        self.status_bar.showMessage(f"Deleted model {info.name}", 3000)

    def on_registry_selection_changed(self) -> None:
        info = self._selected_registry_info()
        if info:
            self.status_bar.showMessage(f"Selected model {info.name}", 2000)

    def on_save_rule_clicked(self) -> None:
        if not self.active_rule:
            QMessageBox.warning(self, "No rule", "Load or create a rule first.")
            return
        try:
            payload = json.loads(self.rule_editor.toPlainText() or "{}")
        except json.JSONDecodeError as exc:
            QMessageBox.critical(self, "Invalid JSON", str(exc))
            return
        with session_scope() as session:
            updated = save_rule(
                session,
                name=self.active_rule.name,
                payload=payload,
                rule_id=self.active_rule.id,
                make_active=True,
            )
        self.active_rule = updated
        self.load_rules_to_combo()
        self.status_bar.showMessage("QC rule saved", 3000)

    def on_new_rule_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "New QC Rule", "Rule name:")
        if not ok or not name.strip():
            return
        try:
            payload = json.loads(self.rule_editor.toPlainText() or "{}")
        except json.JSONDecodeError as exc:
            QMessageBox.critical(self, "Invalid JSON", str(exc))
            return
        with session_scope() as session:
            rule = save_rule(session, name=name.strip(), payload=payload, make_active=True)
        self.active_rule = rule
        self.load_rules_to_combo()
        self.status_bar.showMessage(f"Created rule {rule.name}", 3000)

    def update_rule_editor(self) -> None:
        if not hasattr(self, "rule_editor"):
            return
        payload = {
            "reject_if": self.active_rule.reject_if if self.active_rule else [],
            "min_conf": self.active_rule.min_conf if self.active_rule else 0.0,
        }
        self.rule_editor.blockSignals(True)
        self.rule_editor.setPlainText(json.dumps(payload, indent=2))
        self.rule_editor.blockSignals(False)

    def refresh_settings_roi_list(self) -> None:
        if not hasattr(self, "settings_roi_list"):
            return
        shapes = self.roi_manager.list_presets()
        self.settings_roi_list.blockSignals(True)
        self.settings_roi_list.clear()
        for shape in shapes:
            item = QListWidgetItem(f"{shape.line}: {shape.name} ({shape.kind})")
            item.setData(Qt.UserRole, shape)
            self.settings_roi_list.addItem(item)
        self.settings_roi_list.blockSignals(False)

    def on_settings_add_roi(self) -> None:
        line, ok = QInputDialog.getText(self, "ROI Line", "Line identifier:")
        if not ok or not line.strip():
            return
        name, ok = QInputDialog.getText(self, "ROI Name", "Preset name:")
        if not ok or not name.strip():
            return
        kind, ok = QInputDialog.getItem(self, "ROI Type", "Type", ["rect", "poly"], 0, False)
        if not ok:
            return
        points_text, ok = QInputDialog.getText(
            self,
            "ROI Points",
            "Enter points as x,y;x,y;... (rect requires two points)",
        )
        if not ok or not points_text.strip():
            return
        points: List[Tuple[float, float]] = []
        try:
            for chunk in points_text.split(";"):
                x_str, y_str = chunk.split(",")
                points.append((float(x_str), float(y_str)))
        except ValueError:
            QMessageBox.critical(self, "Invalid input", "Could not parse points. Use format x,y;x,y;...")
            return
        shape = ROIShape(id=None, line=line.strip(), name=name.strip(), kind=kind, points=points)
        self.roi_manager.save(shape)
        self.reload_roi_presets()
        self.refresh_settings_roi_list()
        self.status_bar.showMessage("ROI added", 3000)

    def on_settings_delete_roi(self) -> None:
        items = self.settings_roi_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Select ROI", "Select a preset to delete.")
            return
        shape = items[0].data(Qt.UserRole)
        if not isinstance(shape, ROIShape) or shape.id is None:
            return
        with session_scope() as session:
            preset = session.get(RoiPreset, shape.id)
            if preset:
                session.delete(preset)
        self.reload_roi_presets()
        self.refresh_settings_roi_list()
        self.status_bar.showMessage("ROI deleted", 3000)

    def on_settings_roi_selected(self) -> None:
        items = self.settings_roi_list.selectedItems()
        if not items:
            return
        shape = items[0].data(Qt.UserRole)
        if isinstance(shape, ROIShape):
            self.status_bar.showMessage(f"Selected ROI {shape.line}/{shape.name}", 2000)

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






















































