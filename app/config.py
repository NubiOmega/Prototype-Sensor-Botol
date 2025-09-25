"""Configuration utilities for persisting user preferences."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

DEFAULT_TRAINING: Dict[str, Any] = {
    "data_path": "dataset/data.yaml",
    "model_path": "yolo11n.pt",
    "epochs": 80,
    "imgsz": 960,
    "batch": 16,
    "device": "cpu",
    "workers": 2,
    "resume": False,
    "noval": False,
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera_source": {
        "type": "device",  # "device" or "url"
        "value": 0,
    },
    "stream_url": "http://",
    "model_path": "",
    "confidence": 0.25,
    "selected_classes": [],
    "record_enabled": False,
    "model_choice": "yolo11n.pt",
    "training": DEFAULT_TRAINING,
    "last_trained_weights": "",
}


def load_config() -> Dict[str, Any]:
    """Return stored config or defaults if file is missing/invalid."""
    if not CONFIG_PATH.exists():
        return deepcopy(DEFAULT_CONFIG)
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return deepcopy(DEFAULT_CONFIG)
    merged = deepcopy(DEFAULT_CONFIG)
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "training" and isinstance(value, dict):
                merged["training"].update(value)
            else:
                merged[key] = value
    return merged


def save_config(data: Dict[str, Any]) -> None:
    """Persist configuration to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def update_config(**kwargs: Any) -> Dict[str, Any]:
    """Helper to update and save config in one call."""
    config = load_config()
    for key, value in kwargs.items():
        if key == "training" and isinstance(value, dict):
            config.setdefault("training", deepcopy(DEFAULT_TRAINING))
            config["training"].update(value)
        else:
            config[key] = value
    save_config(config)
    return config

