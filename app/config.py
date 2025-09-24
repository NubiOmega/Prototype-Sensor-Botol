"""Configuration utilities for persisting user preferences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

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
}


def load_config() -> Dict[str, Any]:
    """Return stored config or defaults if file is missing/invalid."""
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG.copy()
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_CONFIG.copy()
    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    return merged


def save_config(data: Dict[str, Any]) -> None:
    """Persist configuration to disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def update_config(**kwargs: Any) -> Dict[str, Any]:
    """Helper to update and save config in one call."""
    config = load_config()
    config.update(kwargs)
    save_config(config)
    return config

