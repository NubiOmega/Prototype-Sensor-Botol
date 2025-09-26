"""Review and relabel UI components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .dataset_tools import LabelBox


@dataclass(slots=True)
class DetectionBox:
    cls: str
    conf: float
    rect: QRectF


class BoxGraphicsItem(QGraphicsRectItem):
    def __init__(self, box: DetectionBox, color: QColor) -> None:
        super().__init__(box.rect)
        self.box = box
        pen = QPen(color, 2)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def to_label(self) -> LabelBox:
        rect = self.rect()
        return LabelBox(
            cls=self.box.cls,
            x1=rect.left(),
            y1=rect.top(),
            x2=rect.right(),
            y2=rect.bottom(),
        )


class ReviewScene(QGraphicsScene):
    selectionChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None

    def load_image(self, pixmap: QPixmap) -> None:
        self.clear()
        self.pixmap_item = self.addPixmap(pixmap)


class ReviewWidget(QWidget):
    saveRequested = Signal()
    exportRequested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        self.view = QGraphicsView()
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.scene = ReviewScene()
        self.view.setScene(self.scene)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        self.list_widget = QListWidget()
        self.load_btn = QPushButton("Load Image")
        self.save_btn = QPushButton("Save Truth")
        self.export_btn = QPushButton("Export Dataset")
        side_layout.addWidget(QLabel("Detections"))
        side_layout.addWidget(self.list_widget)
        side_layout.addWidget(self.load_btn)
        side_layout.addWidget(self.save_btn)
        side_layout.addWidget(self.export_btn)
        side_layout.addStretch(1)

        layout.addWidget(self.view, 3)
        layout.addWidget(side_panel, 1)

        self.save_btn.clicked.connect(self.saveRequested.emit)
        self.export_btn.clicked.connect(self.exportRequested.emit)
        self.load_btn.clicked.connect(self._on_load_image)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.scene.load_image(pixmap)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def load_from_path(self, image_path: Path) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return
        self.set_pixmap(pixmap)

    def _on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select image", str(Path.cwd()))
        if path:
            self.load_from_path(Path(path))


__all__ = [
    "DetectionBox",
    "ReviewWidget",
]

