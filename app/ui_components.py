"""Reusable UI components for the desktop app."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class FilePicker(QWidget):
    valueChanged = Signal(str)

    def __init__(self, title: str, dialog_filter: str, placeholder: str = "") -> None:
        super().__init__()
        self.dialog_title = title
        self.dialog_filter = dialog_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText(placeholder)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._on_browse)
        self.line_edit.textChanged.connect(self.valueChanged.emit)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_btn)

    def setText(self, text: str) -> None:  # noqa: N802
        self.line_edit.setText(text)

    def text(self) -> str:
        return self.line_edit.text()

    def set_read_only(self, read_only: bool) -> None:
        self.line_edit.setReadOnly(read_only)

    def set_button_enabled(self, enabled: bool) -> None:
        self.browse_btn.setEnabled(enabled)

    def _on_browse(self) -> None:
        current_path = self.line_edit.text() or str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, self.dialog_title, current_path, self.dialog_filter)
        if path:
            self.line_edit.setText(path)



class ConfidenceSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum: float = 0.1, maximum: float = 0.9, default: float = 0.25) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(f"Confidence: {default:.2f}")
        self.slider = QSlider(Qt.Horizontal)
        self._precision = 100
        self._minimum = minimum
        self._maximum = maximum
        self.slider.setMinimum(int(minimum * self._precision))
        self.slider.setMaximum(int(maximum * self._precision))
        self.slider.setValue(int(default * self._precision))
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value: int) -> None:
        real_value = value / self._precision
        self.label.setText(f"Confidence: {real_value:.2f}")
        self.valueChanged.emit(real_value)

    def value(self) -> float:
        return self.slider.value() / self._precision

    def setValue(self, value: float) -> None:  # noqa: N802
        self.slider.setValue(int(value * self._precision))


class ClassSelection(QWidget):
    valueChanged = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("Classes")
        self.list_widget = QListWidget()
        layout.addWidget(self.label)
        layout.addWidget(self.list_widget)
        self.list_widget.itemChanged.connect(self._emit_selection)

    def set_classes(self, classes: Iterable[str]) -> None:
        self.list_widget.clear()
        for cls in classes:
            item = QListWidgetItem(cls)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        self._emit_selection()

    def selected(self) -> List[str]:
        selections: List[str] = []
        for index in range(self.list_widget.count()):
            item = self.list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selections.append(item.text())
        return selections

    def _emit_selection(self) -> None:
        self.valueChanged.emit(self.selected())

