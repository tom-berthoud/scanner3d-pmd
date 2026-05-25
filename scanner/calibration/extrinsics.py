"""Utilities for camera extrinsics calibration files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml


def save_camera_extrinsics(
    rotation_matrix: np.ndarray,
    translation_mm: np.ndarray,
    path: str,
    report: dict | None = None,
) -> None:
    """Persist camera-to-platform extrinsics to YAML."""
    data = {
        "rotation_matrix": np.asarray(rotation_matrix, dtype=np.float64).tolist(),
        "translation_mm": np.asarray(translation_mm, dtype=np.float64).reshape(3).tolist(),
    }
    if report is not None:
        data["report"] = report
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False)


def default_extrinsics_path(camera_id: str) -> str:
    """Return the conventional extrinsics file path for a camera id."""
    return str(
        Path(__file__).resolve().parent.parent.parent
        / "config"
        / f"camera_extrinsics_{camera_id}.yaml"
    )
