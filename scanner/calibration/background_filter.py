"""scanner.calibration.background_filter — Persist left-image background masking."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_FILTER_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "background_filter.yaml"


def _default_filter() -> dict[str, Any]:
    return {
        "enabled": False,
        "crop_left_of_col": None,
        "background_line_max_col": None,
        "margin_px": 0,
        "threshold": None,
        "min_pixels": None,
        "extraction_mode": None,
        "captured_at": None,
    }


def load_background_filter(path: str | None = None) -> dict[str, Any]:
    """Load the background-line crop settings from YAML."""
    filter_path = Path(path) if path is not None else _DEFAULT_FILTER_PATH
    if not filter_path.exists():
        return _default_filter()

    with filter_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    data = _default_filter()
    data["enabled"] = bool(raw.get("enabled", False))
    crop_left = raw.get("crop_left_of_col")
    data["crop_left_of_col"] = None if crop_left is None else float(crop_left)
    bg_col = raw.get("background_line_max_col")
    data["background_line_max_col"] = None if bg_col is None else float(bg_col)
    margin_px = raw.get("margin_px")
    data["margin_px"] = 0 if margin_px is None else int(margin_px)
    threshold = raw.get("threshold")
    data["threshold"] = None if threshold is None else int(threshold)
    min_pixels = raw.get("min_pixels")
    data["min_pixels"] = None if min_pixels is None else int(min_pixels)
    extraction_mode = raw.get("extraction_mode")
    data["extraction_mode"] = None if extraction_mode is None else str(extraction_mode)
    data["captured_at"] = raw.get("captured_at")
    return data


def save_background_filter(
    crop_left_of_col: float,
    background_line_max_col: float,
    margin_px: int,
    threshold: int,
    min_pixels: int,
    extraction_mode: str,
    path: str | None = None,
) -> dict[str, Any]:
    """Persist the left-image crop settings to YAML."""
    filter_path = Path(path) if path is not None else _DEFAULT_FILTER_PATH
    data = {
        "enabled": True,
        "crop_left_of_col": float(crop_left_of_col),
        "background_line_max_col": float(background_line_max_col),
        "margin_px": int(margin_px),
        "threshold": int(threshold),
        "min_pixels": int(min_pixels),
        "extraction_mode": str(extraction_mode),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(filter_path.parent, exist_ok=True)
    with filter_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return data


def disable_background_filter(path: str | None = None) -> dict[str, Any]:
    """Disable the current background-line crop while keeping its last values."""
    filter_path = Path(path) if path is not None else _DEFAULT_FILTER_PATH
    data = load_background_filter(path=str(filter_path))
    data["enabled"] = False
    os.makedirs(filter_path.parent, exist_ok=True)
    with filter_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return data
