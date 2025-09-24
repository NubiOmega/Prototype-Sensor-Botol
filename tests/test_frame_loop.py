import sys
import unittest

import numpy as np
from PySide6.QtCore import QCoreApplication

from app.workers import VideoWorker


if QCoreApplication.instance() is None:
    QCoreApplication([])


class MockCamera:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.released = False

    def read(self):
        return self.frame.copy()

    def release(self):
        self.released = True


class MockDetector:
    def predict(self, frame, confidence, class_filter=None):
        return frame, [], 1.0


class FrameLoopTestCase(unittest.TestCase):
    def test_worker_runs_iterations(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        camera = MockCamera(frame)
        detector = MockDetector()
        worker = VideoWorker(camera, detector, max_iterations=5)
        captured = []

        worker.frameReady.connect(lambda f: captured.append(f))
        worker.run()

        self.assertEqual(len(captured), 5)
        self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
