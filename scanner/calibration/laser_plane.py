"""scanner.calibration.laser_plane — Laser sheet plane calibration.

Fits the plane equation ax + by + cz + d = 0 (in camera frame) from
line detections on a flat reference surface at known distances.

Algorithm:
    For each reference image at known distance z_ref:
        1. Extract laser line pixels
        2. Undistort with camera intrinsics
        3. Back-project to 3D using z = z_ref (known ground-truth depth)
    Fit a plane through all collected 3D points using least-squares SVD.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_PLANE_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "config" / "laser_plane.yaml"
)


def calibrate_laser_plane(
    reference_images: list[np.ndarray],
    reference_distances_mm: list[float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Fit the laser sheet plane from flat-surface reference images.

    For each reference image, the laser line is detected and back-projected
    to 3D using the known distance.  A plane is then fitted by SVD.

    Args:
        reference_images: BGR images of the flat reference surface with the
            laser projected on it.  One image per distance.
        reference_distances_mm: Known distances (in mm) from the camera to
            the reference surface for each image.  Must have the same length
            as *reference_images*.
        camera_matrix: 3×3 intrinsic camera matrix.
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3].
        output_path: Destination YAML file path.  Defaults to
            config/laser_plane.yaml in the project root.
    Returns:
        1-D float64 array [a, b, c, d] of the fitted plane equation
        ax + by + cz + d = 0 in the camera coordinate frame.

    Raises:
        CalibrationError: if fewer than 3 valid line detections are available
            or the plane fit fails.
    """
    from scanner.calibration import CalibrationError
    from scanner.processing.laser_line import extract_laser_line
    import cv2  # type: ignore[import]

    if len(reference_images) != len(reference_distances_mm):
        raise CalibrationError(
            f"Mismatch: {len(reference_images)} images but "
            f"{len(reference_distances_mm)} distances"
        )

    all_points: list[np.ndarray] = []

    for img_idx, (img, z_ref) in enumerate(zip(reference_images, reference_distances_mm)):
        line_px = extract_laser_line(img, threshold=20, min_pixels=5, subpixel=True)
        if line_px.shape[0] < 5:
            logger.warning(
                "Image %d: only %d laser pixels found — skipping", img_idx, line_px.shape[0]
            )
            continue

        # Undistort pixels
        pts = line_px.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts, camera_matrix, dist_coeffs).reshape(-1, 2)

        # Back-project to 3D using known z
        # normalised coords (x_n, y_n) → 3D: X = x_n * z, Y = y_n * z, Z = z
        x3d = undist[:, 0] * z_ref
        y3d = undist[:, 1] * z_ref
        z3d = np.full(len(undist), z_ref)
        pts_3d = np.column_stack([x3d, y3d, z3d])
        all_points.append(pts_3d)

        logger.debug(
            "Image %d: %d points at z=%.1f mm", img_idx, len(pts_3d), z_ref
        )

    if len(all_points) < 1:
        raise CalibrationError("No valid laser line detections found in reference images")

    points = np.vstack(all_points)

    if len(points) < 3:
        raise CalibrationError(
            f"Need at least 3 points for plane fitting, got {len(points)}"
        )

    # Fit plane using SVD
    # Centre the points
    centroid = points.mean(axis=0)
    centred = points - centroid

    # SVD — smallest singular value gives normal vector
    _, _, Vt = np.linalg.svd(centred)
    normal = Vt[-1]  # (3,) — normal to best-fit plane

    # Compute d: plane passes through centroid → dot(normal, centroid) + d = 0
    d = -float(normal @ centroid)

    # Normalise so that |normal| = 1
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-12:
        raise CalibrationError("Degenerate plane fit — all points may be collinear")
    normal = normal / norm_len
    d = d / norm_len

    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
    logger.info("Laser plane fitted: [a=%.6f, b=%.6f, c=%.6f, d=%.6f]", *plane)

    # Compute angle between plane normal and camera Z axis (quality check)
    angle_deg = float(
        np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
    )
    logger.info("Plane angle relative to camera Z axis: %.1f° (target ~30°)", angle_deg)

    out_path = output_path or _DEFAULT_PLANE_PATH
    _save_laser_plane(plane, angle_deg, out_path)

    return plane


def _save_laser_plane(
    plane: np.ndarray,
    angle_deg: float,
    path: str,
) -> None:
    """Persist laser plane to YAML.

    Args:
        plane: [a, b, c, d] array.
        angle_deg: Triangulation angle in degrees.
        path: Destination file path.
    """
    data = {
        "plane": {
            "a": float(plane[0]),
            "b": float(plane[1]),
            "c": float(plane[2]),
            "d": float(plane[3]),
        },
        "triangulation_angle_deg": float(angle_deg),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    logger.info("Laser plane calibration saved to %s", path)


def load_laser_plane(
    path: Optional[str] = None,
) -> np.ndarray:
    """Load the laser plane equation from a YAML file.

    Args:
        path: Path to the YAML file.  Defaults to config/laser_plane.yaml.

    Returns:
        Float64 array [a, b, c, d] of the plane equation ax+by+cz+d=0.

    Raises:
        CalibrationError: if the file does not exist or is malformed.
    """
    from scanner.calibration import CalibrationError

    load_path = path or _DEFAULT_PLANE_PATH

    if not os.path.exists(load_path):
        raise CalibrationError(
            f"Laser plane calibration file not found: {load_path}\n"
            "Run: python -m scanner.calibration.laser_plane"
        )

    try:
        with open(load_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        pl = data["plane"]
        plane = np.array(
            [float(pl["a"]), float(pl["b"]), float(pl["c"]), float(pl["d"])],
            dtype=np.float64,
        )
        logger.info(
            "Laser plane loaded from %s: [%.4f, %.4f, %.4f, %.4f]", load_path, *plane
        )
        return plane

    except (KeyError, TypeError) as exc:
        raise CalibrationError(
            f"Laser plane file is malformed ({load_path}): {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise CalibrationError(
            f"YAML parse error in laser plane file ({load_path}): {exc}"
        ) from exc
