import unittest

from app.camera import enumerate_cameras


class CameraEnumTestCase(unittest.TestCase):
    def test_enumerate_returns_list(self):
        devices = enumerate_cameras(max_devices=1)
        self.assertIsInstance(devices, list)
        for device in devices:
            self.assertIsInstance(device, tuple)
            self.assertGreaterEqual(len(device), 2)


if __name__ == "__main__":
    unittest.main()
