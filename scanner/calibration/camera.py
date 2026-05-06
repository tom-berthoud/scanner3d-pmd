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


def _detect_checkerboard(
    image: np.ndarray,
    board_size: tuple[int, int] = (9, 6),
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    """Detect checkerboard corners with the robust SB detector, then fallback."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
            if found and corners is not None:
                return True, corners.astype(np.float32), gray
        except cv2.error:
            logger.debug("findChessboardCornersSB failed, falling back", exc_info=True)

    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners.astype(np.float32), gray
    return False, None, gray


def checkerboard_capture_quality(
    image: np.ndarray,
    board_size: tuple[int, int] = (9, 6),
    previous_poses: Optional[list[list[float]]] = None,
) -> dict:
    """Return detection, exposure and pose-diversity quality for one frame."""
    found, corners, gray = _detect_checkerboard(image, board_size)
    mean = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturated_pct = float(np.mean(gray >= 250) * 100.0)
    dark_pct = float(np.mean(gray <= 5) * 100.0)

    issues: list[str] = []
    if mean < 35.0:
        issues.append("trop sombre")
    if mean > 225.0 or saturated_pct > 8.0:
        issues.append("surexpose")
    if contrast < 18.0:
        issues.append("contraste faible")
    if sharpness < 25.0:
        issues.append("image floue")
    if dark_pct > 30.0:
        issues.append("beaucoup de zones noires")
    if not found:
        issues.append("damier introuvable")

    pose: list[float] | None = None
    pose_distance = None
    if found and corners is not None:
        pts = corners.reshape(-1, 2)
        center = pts.mean(axis=0)
        span = np.ptp(pts, axis=0)
        angle = float(np.degrees(np.arctan2(pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0])))
        pose = [
            float(center[0] / max(gray.shape[1], 1)),
            float(center[1] / max(gray.shape[0], 1)),
            float(span[0] / max(gray.shape[1], 1)),
            float(span[1] / max(gray.shape[0], 1)),
            angle / 180.0,
        ]
        if previous_poses:
            pose_distance = min(float(np.linalg.norm(np.array(pose) - np.array(p))) for p in previous_poses)
            if pose_distance < 0.08:
                issues.append("pose trop similaire")

    accepted = found and not issues
    if found and issues == ["pose trop similaire"]:
        accepted = False

    score = 0.0
    if found:
        score += 50.0
    score += max(0.0, 25.0 - abs(mean - 125.0) / 4.0)
    score += min(20.0, contrast / 3.0)
    score += min(20.0, sharpness / 30.0)
    score -= saturated_pct * 2.0
    if pose_distance is not None:
        score += min(15.0, pose_distance * 80.0)

    return {
        "found": bool(found),
        "accepted": bool(accepted),
        "status": "ok" if accepted else ", ".join(issues),
        "issues": issues,
        "pose": pose,
        "pose_distance": pose_distance,
        "score": float(score),
        "metrics": {
            "brightness_mean": mean,
            "contrast_std": contrast,
            "sharpness": sharpness,
            "saturated_pct": saturated_pct,
            "dark_pct": dark_pct,
        },
        "corners": corners,
    }


def draw_checkerboard_overlay(
    image: np.ndarray,
    board_size: tuple[int, int] = (9, 6),
    quality: Optional[dict] = None,
) -> np.ndarray:
    """Draw checkerboard corners and a compact status overlay."""
    overlay = image.copy()
    quality = quality or checkerboard_capture_quality(image, board_size)
    corners = quality.get("corners")
    found = bool(quality.get("found"))
    if corners is not None:
        cv2.drawChessboardCorners(overlay, board_size, corners, found)

    color = (0, 200, 0) if quality.get("accepted") else (0, 165, 255)
    if not found:
        color = (0, 0, 255)
    text = "damier OK" if quality.get("accepted") else str(quality.get("status", "capture refusee"))
    cv2.rectangle(overlay, (8, 8), (min(overlay.shape[1] - 8, 460), 42), (0, 0, 0), -1)
    cv2.putText(overlay, text[:48], (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return overlay


def calibrate_camera_with_report(
    images: list[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    output_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate camera intrinsics and return a compact quality report."""
    camera_matrix, dist_coeffs, report = _calibrate_camera_impl(
        images, board_size=board_size, square_size_mm=square_size_mm, output_path=output_path
    )
    return camera_matrix, dist_coeffs, report


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
    camera_matrix, dist_coeffs, _report = _calibrate_camera_impl(
        images, board_size=board_size, square_size_mm=square_size_mm, output_path=output_path
    )
    return camera_matrix, dist_coeffs


def _calibrate_camera_impl(
    images: list[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    output_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
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

        found, corners, _gray = _detect_checkerboard(img, board_size)
        if not found:
            logger.debug("Image %d: checkerboard not found", idx)
            continue

        corners_refined = corners
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

    report = {
        "rms_error": float(rms),
        "usable_images": len(object_points),
        "input_images": len(images),
        "image_size": list(image_size),
        "quality": "good" if rms < 0.5 else "ok" if rms < 1.5 else "poor",
    }
    return camera_matrix, dist_coeffs.flatten(), report


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
