"""scanner.calibration.camera — Intrinsic camera calibration using a checkerboard.

Computes the 3×3 camera matrix and distortion coefficients from a set of
checkerboard images using OpenCV's calibrateCamera.  Results are saved to
config/camera_intrinsics.yaml.

IMPORTANT: The output file is config/camera_intrinsics.yaml — never commit
this file with real calibration values (agents.md §5.4).
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2  # type: ignore[import]
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_INTRINSICS_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "config" / "camera_intrinsics.yaml"
)


def calibrate_camera(
    images: list[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    output_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate camera intrinsics from checkerboard images.

    Args:
        images: List of BGR images (numpy arrays) showing the checkerboard
            from different angles/positions.  At least 10 images recommended.
        board_size: (columns, rows) of *inner* corner intersections on the
            checkerboard (default 9×6).
        square_size_mm: Size of one square in millimetres (default 25.0).
        output_path: Path to write the calibration YAML.  Defaults to
            config/camera_intrinsics.yaml relative to the project root.

    Returns:
        Tuple (camera_matrix, dist_coeffs) where:
            - camera_matrix: float64 array of shape (3, 3)
            - dist_coeffs: float64 array of shape (1, 5) — [k1,k2,p1,p2,k3]

    Raises:
        CalibrationError: if fewer than 4 usable checkerboard images are found
            or if OpenCV calibration fails to converge.
    """
    from scanner.calibration import CalibrationError

    obj_cols, obj_rows = board_size
    # 3D object points for one checkerboard image (z=0 plane)
    objp = np.zeros((obj_rows * obj_cols, 3), dtype=np.float64)
    objp[:, :2] = (
        np.mgrid[0:obj_cols, 0:obj_rows].T.reshape(-1, 2) * square_size_mm
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: Optional[tuple[int, int]] = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    logger.info(
        "Searching for %dx%d checkerboard corners in %d images",
        obj_cols,
        obj_rows,
        len(images),
    )

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            logger.debug("Image %d: checkerboard not found", idx)
            continue

        # Refine corner positions to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        object_points.append(objp)
        image_points.append(corners_refined)
        logger.debug("Image %d: %d corners found", idx, len(corners_refined))

    if len(object_points) < 4:
        raise CalibrationError(
            f"Only {len(object_points)} usable checkerboard images found "
            f"(minimum 4 required).  Make sure the checkerboard is fully "
            f"visible and well-lit."
        )

    logger.info(
        "Running calibrateCamera with %d image pairs", len(object_points)
    )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,  # type: ignore[arg-type]
        None,
        None,
    )

    if rms > 2.0:
        logger.warning(
            "Camera calibration RMS reprojection error is %.4f px "
            "(target < 0.5 px).  Consider adding more/better images.",
            rms,
        )
    else:
        logger.info("Camera calibration RMS reprojection error: %.4f px", rms)

    # Save to YAML
    out_path = output_path or _DEFAULT_INTRINSICS_PATH
    _save_camera_calibration(camera_matrix, dist_coeffs, image_size, rms, out_path)

    return camera_matrix, dist_coeffs.flatten()


def _save_camera_calibration(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    path: str,
) -> None:
    """Persist calibration results to a YAML file.

    Args:
        camera_matrix: (3, 3) camera intrinsic matrix.
        dist_coeffs: distortion coefficients array.
        image_size: (width, height) used during calibration.
        rms: RMS reprojection error.
        path: Destination file path.
    """
    dc = dist_coeffs.flatten().tolist()
    data = {
        "camera_matrix": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
        "dist_coeffs": dc if len(dc) >= 5 else dc + [0.0] * (5 - len(dc)),
        "image_size": list(image_size),
        "rms_error": float(rms),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    logger.info("Camera calibration saved to %s", path)


def load_camera_calibration(
    path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics from a YAML file.

    Args:
        path: Path to the YAML file.  Defaults to
            config/camera_intrinsics.yaml in the project root.

    Returns:
        Tuple (camera_matrix, dist_coeffs) where:
            - camera_matrix: float64 array of shape (3, 3)
            - dist_coeffs: float64 array of shape (5,) — [k1,k2,p1,p2,k3]

    Raises:
        CalibrationError: if the file does not exist or is malformed.
    """
    from scanner.calibration import CalibrationError

    load_path = path or _DEFAULT_INTRINSICS_PATH

    if not os.path.exists(load_path):
        raise CalibrationError(
            f"Camera calibration file not found: {load_path}\n"
            "Run: python -m scanner.calibration.camera"
        )

    try:
        with open(load_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        cm = data["camera_matrix"]
        fx, fy = float(cm["fx"]), float(cm["fy"])
        cx, cy = float(cm["cx"]), float(cm["cy"])
        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

        dc = data["dist_coeffs"]
        dist_coeffs = np.array(dc, dtype=np.float64).flatten()

        logger.info("Camera calibration loaded from %s (RMS=%.4f)", load_path, data.get("rms_error", 0))
        return camera_matrix, dist_coeffs

    except (KeyError, TypeError) as exc:
        raise CalibrationError(
            f"Camera calibration file is malformed ({load_path}): {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise CalibrationError(
            f"YAML parse error in calibration file ({load_path}): {exc}"
        ) from exc


def approximate_camera_intrinsics(
    resolution: tuple[int, int],
    focal_scale: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Build approximate intrinsics for runtime without checkerboard calibration.

    Args:
        resolution: (width, height) camera resolution in pixels.
        focal_scale: Multiplier applied to width to estimate fx/fy.
            With the default 1.25, 640 px => 800 px focal length.

    Returns:
        Tuple (camera_matrix, dist_coeffs) as float64 numpy arrays.
    """
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid resolution for approximate intrinsics: {resolution}")
    if focal_scale <= 0:
        raise ValueError(f"focal_scale must be > 0, got {focal_scale}")

    fx = float(width) * float(focal_scale)
    fy = float(width) * float(focal_scale)
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)

    logger.warning(
        "Using approximate camera intrinsics (checkerboard disabled): "
        "fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
        fx,
        fy,
        cx,
        cy,
    )
    return camera_matrix, dist_coeffs
