"""scanner.calibration.laser_plane — Laser sheet plane calibration.

Fits the plane equation ax + by + cz + d = 0 from line detections on a flat
reference surface at known distances. The current scan pipeline uses one
shared laser plane in the platform frame.

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


def _undistort_to_normalized(
    line_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    dist = np.asarray(dist_coeffs, dtype=np.float64).flatten()
    if dist.size == 0 or np.allclose(dist, 0.0):
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        return np.column_stack(
            [
                (line_px[:, 0].astype(np.float64) - cx) / fx,
                (line_px[:, 1].astype(np.float64) - cy) / fy,
            ]
        )

    import cv2  # type: ignore[import]

    pts = line_px.reshape(-1, 1, 2).astype(np.float64)
    return cv2.undistortPoints(pts, camera_matrix, dist_coeffs).reshape(-1, 2)


def calibrate_laser_plane(
    reference_images: list[np.ndarray],
    reference_distances_mm: list[float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    output_path: Optional[str] = None,
    threshold: int = 20,
    min_pixels: int = 5,
    mode: str = "row_mean",
    mask_rects: list | None = None,
    camera_id: str | None = None,
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

    if len(reference_images) != len(reference_distances_mm):
        raise CalibrationError(
            f"Mismatch: {len(reference_images)} images but "
            f"{len(reference_distances_mm)} distances"
        )

    all_points: list[np.ndarray] = []

    for img_idx, (img, z_ref) in enumerate(zip(reference_images, reference_distances_mm)):
        line_px = extract_laser_line(
            img,
            threshold=threshold,
            min_pixels=min_pixels,
            subpixel=True,
            mode=mode,
            camera_id=camera_id,
            mask_rects=mask_rects,
        )
        if line_px.shape[0] < 5:
            logger.warning(
                "Image %d: only %d laser pixels found — skipping", img_idx, line_px.shape[0]
            )
            continue

        undist = _undistort_to_normalized(line_px, camera_matrix, dist_coeffs)

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


def calibrate_laser_plane_platform_z(
    reference_images: list[np.ndarray],
    platform_z_mm: list[float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_to_platform_rotation: np.ndarray,
    camera_to_platform_translation: np.ndarray,
    output_path: Optional[str] = None,
    threshold: int = 20,
    min_pixels: int = 5,
    mode: str = "row_mean",
    mask_rects: list | None = None,
    camera_id: str | None = None,
) -> np.ndarray:
    """Fit the laser plane using vertical reference boards at known platform Z.

    The reference board is assumed to be a vertical X/Y plane in the shared
    platform frame, with equation ``Z = platform_z_mm``. This lets the user keep
    the board vertical and facing the laser instead of orienting it toward each
    camera. The fitted laser plane is still saved in the selected camera frame.
    """
    from scanner.calibration import CalibrationError
    from scanner.processing.laser_line import extract_laser_line

    if len(reference_images) != len(platform_z_mm):
        raise CalibrationError(
            f"Mismatch: {len(reference_images)} images but {len(platform_z_mm)} Z positions"
        )

    rot = np.asarray(camera_to_platform_rotation, dtype=np.float64)
    trans = np.asarray(camera_to_platform_translation, dtype=np.float64).reshape(3)
    if rot.shape != (3, 3):
        raise CalibrationError(f"camera_to_platform_rotation must be 3x3, got {rot.shape}")

    all_points: list[np.ndarray] = []
    for img_idx, (img, z_ref) in enumerate(zip(reference_images, platform_z_mm)):
        line_px = extract_laser_line(
            img,
            threshold=threshold,
            min_pixels=min_pixels,
            subpixel=True,
            mode=mode,
            camera_id=camera_id,
            mask_rects=mask_rects,
        )
        if line_px.shape[0] < min_pixels:
            logger.warning(
                "Image %d: only %d laser pixels found - skipping", img_idx, line_px.shape[0]
            )
            continue

        undist = _undistort_to_normalized(line_px, camera_matrix, dist_coeffs)
        rays_cam = np.hstack([undist, np.ones((len(undist), 1), dtype=np.float64)])
        rays_platform = (rot @ rays_cam.T).T
        denom = rays_platform[:, 2]
        valid = np.abs(denom) > 1e-9
        scale = np.full(len(rays_cam), np.nan, dtype=np.float64)
        scale[valid] = (float(z_ref) - trans[2]) / denom[valid]
        valid &= scale > 0.0
        if int(valid.sum()) < min_pixels:
            logger.warning("Image %d: too few forward intersections - skipping", img_idx)
            continue

        all_points.append(rays_cam[valid] * scale[valid, np.newaxis])
        logger.debug("Image %d: %d points at platform Z=%.1f mm", img_idx, int(valid.sum()), z_ref)

    if len(all_points) < 1:
        raise CalibrationError("No valid laser line detections found in reference images")

    points = np.vstack(all_points)
    if len(points) < 3:
        raise CalibrationError(f"Need at least 3 points for plane fitting, got {len(points)}")

    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred)
    normal = vt[-1]
    d = -float(normal @ centroid)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-12:
        raise CalibrationError("Degenerate plane fit - all points may be collinear")
    normal = normal / norm_len
    d = d / norm_len

    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
    out_path = output_path or _DEFAULT_PLANE_PATH
    _save_laser_plane(plane, angle_deg, out_path)
    logger.info("Laser plane fitted from platform-Z boards: [%.6f, %.6f, %.6f, %.6f]", *plane)
    return plane


def collect_laser_points_platform_z(
    reference_images: list[np.ndarray],
    platform_z_mm: list[float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_to_platform_rotation: np.ndarray,
    camera_to_platform_translation: np.ndarray,
    threshold: int = 20,
    min_pixels: int = 5,
    mode: str = "row_mean",
    mask_rects: list | None = None,
    camera_id: str | None = None,
) -> np.ndarray:
    """Back-project laser detections onto known platform ``Z`` boards.

    Returns points in the shared platform frame. These points lie on both the
    reference board and the physical laser sheet, so they can be merged across
    cameras to fit one global laser plane.
    """
    from scanner.calibration import CalibrationError
    from scanner.processing.laser_line import extract_laser_line

    if len(reference_images) != len(platform_z_mm):
        raise CalibrationError(
            f"Mismatch: {len(reference_images)} images but {len(platform_z_mm)} Z positions"
        )

    rot = np.asarray(camera_to_platform_rotation, dtype=np.float64)
    trans = np.asarray(camera_to_platform_translation, dtype=np.float64).reshape(3)
    if rot.shape != (3, 3):
        raise CalibrationError(f"camera_to_platform_rotation must be 3x3, got {rot.shape}")

    all_points: list[np.ndarray] = []
    for img_idx, (img, z_ref) in enumerate(zip(reference_images, platform_z_mm)):
        line_px = extract_laser_line(
            img,
            threshold=threshold,
            min_pixels=min_pixels,
            subpixel=True,
            mode=mode,
            camera_id=camera_id,
            mask_rects=mask_rects,
        )
        if line_px.shape[0] < min_pixels:
            logger.warning(
                "Image %d: only %d laser pixels found - skipping", img_idx, line_px.shape[0]
            )
            continue

        undist = _undistort_to_normalized(line_px, camera_matrix, dist_coeffs)
        rays_cam = np.hstack([undist, np.ones((len(undist), 1), dtype=np.float64)])
        rays_platform = (rot @ rays_cam.T).T
        denom = rays_platform[:, 2]
        valid = np.abs(denom) > 1e-9
        scale = np.full(len(rays_cam), np.nan, dtype=np.float64)
        scale[valid] = (float(z_ref) - trans[2]) / denom[valid]
        valid &= scale > 0.0
        if int(valid.sum()) < min_pixels:
            logger.warning("Image %d: too few forward intersections - skipping", img_idx)
            continue

        points_platform = trans + rays_platform[valid] * scale[valid, np.newaxis]
        all_points.append(points_platform)
        logger.debug("Image %d: %d points at platform Z=%.1f mm", img_idx, int(valid.sum()), z_ref)

    if len(all_points) < 1:
        raise CalibrationError("No valid laser line detections found in reference images")

    points = np.vstack(all_points)
    if len(points) < 3:
        raise CalibrationError(f"Need at least 3 points for plane fitting, got {len(points)}")
    return points


def fit_laser_plane_points(
    points: np.ndarray,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Fit and save one laser plane from platform-frame points."""
    from scanner.calibration import CalibrationError

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CalibrationError(f"points must be (N, 3), got {points.shape}")
    if len(points) < 3:
        raise CalibrationError(f"Need at least 3 points for plane fitting, got {len(points)}")

    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred)
    normal = vt[-1]
    d = -float(normal @ centroid)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-12:
        raise CalibrationError("Degenerate plane fit - all points may be collinear")
    normal = normal / norm_len
    d = d / norm_len

    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
    out_path = output_path or _DEFAULT_PLANE_PATH
    _save_laser_plane(plane, angle_deg, out_path, frame="platform")
    logger.info("Global laser plane fitted in platform frame: [%.6f, %.6f, %.6f, %.6f]", *plane)
    return plane


def calibrate_laser_plane_global_platform_z(
    camera_observations: list[dict],
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Fit one shared platform-frame laser plane from multiple cameras.

    Each observation dict must contain the arguments accepted by
    ``collect_laser_points_platform_z``: images, z_values, camera matrix,
    distortion coefficients, and camera-to-platform extrinsics.
    """
    from scanner.calibration import CalibrationError

    all_points: list[np.ndarray] = []
    for obs in camera_observations:
        points = collect_laser_points_platform_z(
            obs["reference_images"],
            obs["platform_z_mm"],
            obs["camera_matrix"],
            obs["dist_coeffs"],
            obs["camera_to_platform_rotation"],
            obs["camera_to_platform_translation"],
            threshold=int(obs.get("threshold", 20)),
            min_pixels=int(obs.get("min_pixels", 5)),
            mode=str(obs.get("mode", "row_mean")),
            mask_rects=obs.get("mask_rects"),
            camera_id=obs.get("camera_id"),
        )
        all_points.append(points)

    if not all_points:
        raise CalibrationError("No valid laser points found")
    return fit_laser_plane_points(np.vstack(all_points), output_path=output_path)


def _save_laser_plane(
    plane: np.ndarray,
    angle_deg: float,
    path: str,
    frame: str = "camera",
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
        "frame": frame,
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
