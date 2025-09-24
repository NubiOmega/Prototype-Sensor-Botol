"""Camera enumeration and capture management."""
from __future__ import annotations

import logging
from typing import List, Tuple, Union

import cv2

LOGGER = logging.getLogger(__name__)

CameraIdentifier = Union[int, str]


def enumerate_cameras(max_devices: int = 10) -> List[Tuple[int, str]]:
    """Return list of (device_index, friendly_name)."""
    devices: List[Tuple[int, str]] = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            continue
        name = _get_device_name(cap, index)
        devices.append((index, name))
        cap.release()
    return devices


def _get_device_name(capture: cv2.VideoCapture, index: int) -> str:
    """Attempt to get device name, fall back to generic label."""
    name = ""
    # CAP_PROP_POS_M - placeholders; actual name property is backend dependent
    # 270 corresponds to CAP_PROP_WIN32_DEVICE_NAME on some builds.
    name_prop_id = getattr(cv2, "CAP_PROP_WIN32_DEVICE_NAME", 270)
    try:
        value = capture.get(name_prop_id)
        if isinstance(value, str) and value:
            name = value
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Failed to read camera name for index %s", index)
    if not name:
        name = f"Camera {index}"
    return name


class CameraManager:
    """Wrapper around cv2.VideoCapture that supports device indices or URLs."""

    def __init__(self, source: CameraIdentifier):
        self.source = source
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> bool:
        if isinstance(self.source, int):
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.source)
        if not capture or not capture.isOpened():
            LOGGER.error("Failed to open camera source %s", self.source)
            self.capture = None
            return False
        # Reduce latency by shrinking buffer where supported
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture
        return True

    def read(self):
        if not self.capture:
            return None
        ok, frame = self.capture.read()
        if not ok:
            LOGGER.warning("Camera returned no frame")
            return None
        return frame

    def release(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None

    def is_open(self) -> bool:
        return bool(self.capture and self.capture.isOpened())
