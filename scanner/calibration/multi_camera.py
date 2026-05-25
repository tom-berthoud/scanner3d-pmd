"""Utilities for multi-camera scanner configuration and calibration loading."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scanner.calibration.camera import approximate_camera_intrinsics, load_camera_calibration
from scanner.calibration.laser_plane import load_laser_plane

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


def camera_configs(config: dict) -> list[dict]:
    """Return normalized camera configuration entries.

    The new format is ``cameras: [...]``.  If absent, the legacy ``camera:``
    section is exposed as one camera with id ``main``.
    """
    cameras = config.get("cameras")
    if isinstance(cameras, list) and cameras:
        result = []
        for idx, raw in enumerate(cameras):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item.setdefault("id", f"camera_{idx}")
            item.setdefault("type", "pi" if idx == 0 else "usb")
            result.append(item)
        if result:
            return result

    legacy = dict(config.get("camera", {}))
    legacy.setdefault("id", "main")
    legacy.setdefault("type", "pi")
    return [legacy]


def default_camera_id(config: dict) -> str:
    """Return the first configured camera id."""
    return str(camera_configs(config)[0]["id"])


def camera_config_by_id(config: dict, camera_id: str) -> dict:
    """Return one normalized camera config by id."""
    for cam_cfg in camera_configs(config):
        if str(cam_cfg.get("id")) == str(camera_id):
            return cam_cfg
    raise KeyError(f"Unknown camera id: {camera_id}")


def camera_ids(config: dict) -> list[str]:
    """Return configured camera ids in acquisition order."""
    return [str(cam_cfg["id"]) for cam_cfg in camera_configs(config)]


def _project_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return str(_CONFIG_DIR.parent / path)


def _normalize(vec: np.ndarray, name: str) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        raise ValueError(f"{name} vector is degenerate")
    return vec / length


def _look_at_extrinsics(extr: dict) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(extr.get("position_mm"), dtype=np.float64).reshape(3)
    target = np.asarray(extr.get("target_mm", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    up = np.asarray(extr.get("up_mm", [0.0, 1.0, 0.0]), dtype=np.float64).reshape(3)

    forward = _normalize(target - position, "camera forward")
    up = _normalize(up, "camera up")
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = _normalize(right, "camera right")
    down = _normalize(np.cross(forward, right), "camera down")

    rotation = np.column_stack([right, down, forward])
    return rotation, position


def _load_extrinsics(cam_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    extr = cam_cfg.get("extrinsics") or {}
    if not isinstance(extr, dict):
        extr = {}

    path = _project_path(extr.get("path") or cam_cfg.get("extrinsics_path"))
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            file_data = yaml.safe_load(fh) or {}
        if isinstance(file_data, dict):
            extr = {**extr, **file_data}

    if "position_mm" in extr:
        return _look_at_extrinsics(extr)

    rot_raw: Any = extr.get("rotation_matrix", np.eye(3).tolist())
    trans_raw: Any = extr.get("translation_mm", [0.0, 0.0, 0.0])
    rotation = np.asarray(rot_raw, dtype=np.float64)
    translation = np.asarray(trans_raw, dtype=np.float64).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"extrinsics.rotation_matrix must be 3x3, got {rotation.shape}")
    return rotation, translation


def load_camera_model(
    config: dict,
    camera_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load intrinsics, laser plane and extrinsics for one camera.

    Returns:
        ``(camera_matrix, dist_coeffs, laser_plane, rotation, translation)``.
        ``rotation`` and ``translation`` transform camera-frame points into the
        scanner's stationary platform frame before turntable unrotation.
    """
    cam_cfg = camera_config_by_id(config, camera_id)
    calib_cfg = config.get("calibration", {})
    use_checkerboard = bool(calib_cfg.get("use_checkerboard", True))
    focal_scale = float(cam_cfg.get("approx_focal_scale", calib_cfg.get("approx_focal_scale", 1.25)))
    resolution = cam_cfg.get("resolution", config.get("camera", {}).get("resolution", [640, 480]))
    cam_res = (int(resolution[0]), int(resolution[1]))

    intrinsics_path = _project_path(cam_cfg.get("intrinsics_path"))
    if intrinsics_path and not os.path.exists(intrinsics_path):
        logger.warning("Camera intrinsics file for %s not found, falling back", camera_id)
        intrinsics_path = None
    if use_checkerboard:
        camera_matrix, dist_coeffs = load_camera_calibration(intrinsics_path)
    else:
        camera_matrix, dist_coeffs = approximate_camera_intrinsics(cam_res, focal_scale=focal_scale)

    laser_plane_path = _project_path(
        calib_cfg.get("global_laser_plane_path")
        or config.get("laser", {}).get("plane_path")
        or cam_cfg.get("laser_plane_path")
    )
    if laser_plane_path and not os.path.exists(laser_plane_path):
        logger.warning("Laser plane file for %s not found, falling back", camera_id)
        laser_plane_path = None
    laser_plane = load_laser_plane(laser_plane_path)
    rotation, translation = _load_extrinsics(cam_cfg)

    logger.info("Loaded camera model for %s", camera_id)
    return camera_matrix, dist_coeffs, laser_plane, rotation, translation


__all__ = [
    "camera_configs",
    "camera_config_by_id",
    "camera_ids",
    "default_camera_id",
    "load_camera_model",
]
