"""Generate a Google Colab training notebook template."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(slots=True)
class ColabConfig:
    dataset_path: Path
    output_dir: Path
    base_model: str = "yolo11n.pt"
    epochs: int = 80
    imgsz: int = 960
    batch: int = 16


NOTEBOOK_TEMPLATE: Dict = {
    "cells": [],
    "metadata": {
        "colab": {"name": "QC_Training.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def _markdown_cell(text: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(code: str) -> Dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}


def generate_notebook(config: ColabConfig, notebook_path: Path) -> Path:
    dataset_zip = config.output_dir / "dataset_reviewed.zip"
    cells: List[Dict] = [
        _markdown_cell("# QC Training Notebook\nThis notebook trains a YOLO model on the reviewed dataset."),
        _code_cell(
            "# Mount Google Drive\nfrom google.colab import drive\ndrive.mount('/content/drive')"
        ),
        _code_cell(
            "# Install dependencies\n%pip install ultralytics pandas"
        ),
        _code_cell(
            "# Paths\nfrom pathlib import Path\nbase_dir = Path('/content')\ndataset_zip = Path('{zip_path}')\nmodels_dir = base_dir / 'models'\nmodels_dir.mkdir(exist_ok=True)\n".format(zip_path=str(dataset_zip).replace('\\', '/'))
        ),
        _code_cell(
            "# Unzip dataset if needed\nimport zipfile\nif dataset_zip.exists():\n    with zipfile.ZipFile(dataset_zip, 'r') as zf:\n        zf.extractall('/content/qc_dataset')\nelse:\n    print('Upload dataset ZIP to', dataset_zip)"
        ),
        _code_cell(
            "# Training\nfrom ultralytics import YOLO\nmodel = YOLO('{model_path}')\nmodel.train(data='/content/qc_dataset/data.yaml', epochs={epochs}, imgsz={imgsz}, batch={batch})".format(
                model_path=str(config.base_model), epochs=config.epochs, imgsz=config.imgsz, batch=config.batch
            )
        ),
        _code_cell(
            "# Save best weights back to Drive\nimport shutil\nweights = Path('runs/detect/train/weights/best.pt')\nif weights.exists():\n    target = models_dir / 'best.pt'\n    shutil.copy(weights, target)\n    print('Saved best weights to', target)\nelse:\n    print('Weights not found')"
        ),
    ]
    notebook = NOTEBOOK_TEMPLATE.copy()
    notebook["cells"] = cells
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


__all__ = [
    "ColabConfig",
    "generate_notebook",
]
