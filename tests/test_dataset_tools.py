import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

try:
    from dataset_tools import check_dataset, split_folder
except SystemExit:
    check_dataset = None
    split_folder = None


@unittest.skipUnless(check_dataset and split_folder, "PyYAML is required for dataset_tools tests")
class DatasetToolsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.base = Path(self.temp_dir.name)

    def _create_dummy_file(self, path: Path, content: bytes = b"test") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_split_folder_creates_expected_structure(self) -> None:
        images_dir = self.base / "images_src"
        labels_dir = self.base / "labels_src"
        images_dir.mkdir()
        labels_dir.mkdir()
        for idx in range(4):
            self._create_dummy_file(images_dir / f"sample_{idx}.jpg")
            self._create_dummy_file(labels_dir / f"sample_{idx}.txt", b"0 0.5 0.5 1.0 1.0")

        summary = split_folder(images_dir, labels_dir, out_dir=self.base / "dataset", train_ratio=0.5, seed=1)

        self.assertEqual(summary["missing_labels"], 0)
        self.assertEqual(summary["orphan_labels"], 0)
        self.assertEqual(summary["train"], 2)
        self.assertEqual(summary["val"], 2)

        for split in ("train", "val"):
            self.assertTrue((self.base / "dataset" / "images" / split).exists())
            self.assertTrue((self.base / "dataset" / "labels" / split).exists())

    def test_check_dataset_flags_missing_pairs(self) -> None:
        dataset_root = self.base / "dataset"
        (dataset_root / "images" / "train").mkdir(parents=True)
        (dataset_root / "images" / "val").mkdir(parents=True)
        (dataset_root / "labels" / "train").mkdir(parents=True)
        (dataset_root / "labels" / "val").mkdir(parents=True)

        self._create_dummy_file(dataset_root / "images" / "train" / "img1.jpg")
        self._create_dummy_file(dataset_root / "labels" / "train" / "img1.txt", b"0 0.5 0.5 1 1")
        self._create_dummy_file(dataset_root / "images" / "train" / "img2.jpg")

        data_yaml = dataset_root / "data.yaml"
        data_yaml.write_text(
            """path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: defect\n""".format(root=dataset_root.as_posix()),
            encoding="utf-8",
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ok = check_dataset(str(data_yaml))
        output = buffer.getvalue()

        self.assertFalse(ok)
        self.assertIn("Missing label files", output)


if __name__ == "__main__":
    unittest.main()
