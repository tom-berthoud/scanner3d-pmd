"""Extrinsic camera calibration against the scanner platform frame."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from scanner.calibration.camera import _detect_checkerboard


def _platform_board_points(
    board_size: tuple[int, int],
    square_size_mm: float,
    origin_mm: np.ndarray,
    col_axis: np.ndarray,
    row_axis: np.ndarray,
) -> np.ndarray:
    cols, rows = board_size
    col_unit = np.asarray(col_axis, dtype=np.float64).reshape(3)
    row_unit = np.asarray(row_axis, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_mm, dtype=np.float64).reshape(3)
    col_norm = float(np.linalg.norm(col_unit))
    row_norm = float(np.linalg.norm(row_unit))
    if col_norm < 1e-9 or row_norm < 1e-9:
        raise ValueError("Board axes must be non-zero")
    col_unit = col_unit / col_norm
    row_unit = row_unit / row_norm

    points = []
    for row in range(rows):
        for col in range(cols):
            points.append(origin + col_unit * col * square_size_mm + row_unit * row * square_size_mm)
    return np.asarray(points, dtype=np.float32)


def _extrinsics_report(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> dict:
    import cv2  # type: ignore[import]

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    image_points = image_points.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    return {
        "mean_reprojection_error_px": float(errors.mean()),
        "max_reprojection_error_px": float(errors.max()),
        "points": int(len(object_points)),
    }


def calibrate_camera_extrinsics(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_size: tuple[int, int],
    square_size_mm: float,
    board_origin_mm: list[float] | tuple[float, float, float] | np.ndarray,
    board_col_axis: list[float] | tuple[float, float, float] | np.ndarray,
    board_row_axis: list[float] | tuple[float, float, float] | np.ndarray,
    output_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate camera-to-platform extrinsics from one known checkerboard pose.

    The checkerboard object points are expressed directly in the platform
    frame. OpenCV estimates platform-to-camera pose, which is inverted before
    saving because the scan pipeline needs camera-to-platform transforms.
    """
    import cv2  # type: ignore[import]

    from scanner.calibration import CalibrationError

    found, corners, _gray = _detect_checkerboard(image, board_size)
    if not found or corners is None:
        raise CalibrationError("Checkerboard not found in extrinsics image")

    object_points = _platform_board_points(
        board_size,
        square_size_mm,
        np.asarray(board_origin_mm, dtype=np.float64),
        np.asarray(board_col_axis, dtype=np.float64),
        np.asarray(board_row_axis, dtype=np.float64),
    )
    image_points = corners.reshape(-1, 2).astype(np.float32)
    if len(object_points) != len(image_points):
        raise CalibrationError(
            f"Detected {len(image_points)} corners but expected {len(object_points)}"
        )

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise CalibrationError("OpenCV solvePnP failed for camera extrinsics")

    rot_platform_to_cam, _ = cv2.Rodrigues(rvec)
    rot_camera_to_platform = rot_platform_to_cam.T
    trans_camera_to_platform = (-rot_camera_to_platform @ tvec.reshape(3)).astype(np.float64)
    report = _extrinsics_report(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
    )
    report["camera_position_mm"] = trans_camera_to_platform.tolist()
    report["board_origin_mm"] = [float(v) for v in board_origin_mm]
    report["board_col_axis"] = [float(v) for v in board_col_axis]
    report["board_row_axis"] = [float(v) for v in board_row_axis]

    if output_path:
        save_camera_extrinsics(
            rot_camera_to_platform,
            trans_camera_to_platform,
            output_path,
            report=report,
        )
    return rot_camera_to_platform, trans_camera_to_platform, report


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
