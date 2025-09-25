import unittest

from train_defects import parse_args


class TrainArgsTestCase(unittest.TestCase):
    def test_default_arguments(self) -> None:
        args = parse_args([])
        self.assertEqual(args.model, "yolo11n.pt")
        self.assertEqual(args.data, "dataset/data.yaml")
        self.assertEqual(args.epochs, 80)
        self.assertEqual(args.imgsz, 960)
        self.assertEqual(args.batch, 16)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.project, "runs/detect")
        self.assertEqual(args.name, "train")
        self.assertFalse(args.resume)
        self.assertFalse(args.noval)

    def test_device_argument(self) -> None:
        args = parse_args(["--device", "0", "--noval", "--resume"])
        self.assertEqual(args.device, "0")
        self.assertTrue(args.noval)
        self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()
