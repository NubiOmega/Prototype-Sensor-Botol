import unittest

from app.detector import Detector


class ModelLoadTestCase(unittest.TestCase):
    def test_default_model_loads_or_skips(self):
        detector = Detector()
        try:
            classes = detector.load()
        except Exception as exc:  # pragma: no cover - network/offline guard
            self.skipTest(f"Default model unavailable: {exc}")
        self.assertGreater(len(classes), 0)


if __name__ == "__main__":
    unittest.main()
