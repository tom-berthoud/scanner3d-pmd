"""scanner.processing.triangulation — Laser triangulation pixel → 3D.

Converts a detected laser line (pixel coordinates) into 3D world points
using the camera intrinsic parameters and the laser plane equation.

Pipeline:
    1. Undistort pixel coordinates using camera_matrix and dist_coeffs.
    2. Compute normalised camera rays for each pixel.
    3. Find the intersection of each ray with the laser plane ax+by+cz+d=0
       (plane defined in camera frame).
    4. Rotate the 3D points by rotation_angle_rad around the Y axis to
       bring them into the world (object) frame.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def triangulate(
    line_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    laser_plane: np.ndarray,
    rotation_angle_rad: float,
    axis_point: np.ndarray | None = None,
) -> np.ndarray:
    """Convert laser line pixels to 3D world points via triangulation.

    Args:
        line_pixels: Float array of shape (N, 2) with [col, row] coordinates
            as returned by extract_laser_line.
        camera_matrix: 3×3 intrinsic camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]].
        dist_coeffs: Distortion coefficient array [k1, k2, p1, p2, k3].
        laser_plane: 1-D array [a, b, c, d] defining the laser sheet plane
            as ax + by + cz + d = 0 in the camera coordinate frame.
        rotation_angle_rad: Current rotation angle of the turntable (radians).
            Used to rotate camera-frame points into the world frame.
        axis_point: Optional 1-D array [x, y, z] (mm) of a point on the
            rotation axis in the camera frame (typically the turntable centre).
            When provided the rotation is applied as:
            ``P_world = Ry(-θ) @ (P_cam − axis_point)``
            which places the world-frame origin at the turntable centre.
            When ``None`` the rotation is applied around the camera origin.

    Returns:
        Float array of shape (N, 3) with (X, Y, Z) world coordinates in mm.
        Returns empty array of shape (0, 3) if *line_pixels* is empty.

    Raises:
        ValueError: if array shapes are inconsistent.
    """
    if line_pixels.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    if line_pixels.ndim != 2 or line_pixels.shape[1] != 2:
        raise ValueError(f"line_pixels must be (N, 2), got {line_pixels.shape}")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be (3, 3), got {camera_matrix.shape}")
    if laser_plane.shape != (4,):
        raise ValueError(f"laser_plane must be (4,), got {laser_plane.shape}")

    import cv2  # type: ignore[import]

    n_points = line_pixels.shape[0]

    # ------------------------------------------------------------------ #
    # Step 1 — Undistort pixel coordinates
    # ------------------------------------------------------------------ #
    # cv2.undistortPoints expects shape (N, 1, 2)
    pts_distorted = line_pixels.reshape(-1, 1, 2).astype(np.float64)
    pts_undistorted = cv2.undistortPoints(
        pts_distorted, camera_matrix, dist_coeffs
    )
    # Result shape (N, 1, 2) in normalised image coordinates (after removing K)

    # ------------------------------------------------------------------ #
    # Step 2 — Build camera rays in camera frame
    # ------------------------------------------------------------------ #
    # Normalised coordinates (x_n, y_n) already have camera matrix removed.
    # Ray direction in camera frame: d = [x_n, y_n, 1]
    xy_norm = pts_undistorted.reshape(-1, 2)  # (N, 2)
    ones = np.ones((n_points, 1), dtype=np.float64)
    rays = np.hstack([xy_norm, ones])  # (N, 3) — unit direction vectors

    # ------------------------------------------------------------------ #
    # Step 3 — Intersect rays with laser plane  ax+by+cz+d = 0
    # ------------------------------------------------------------------ #
    a, b, c, d = laser_plane.astype(np.float64)
    plane_normal = np.array([a, b, c], dtype=np.float64)

    # Ray: P = t * ray_dir (origin at camera centre = [0,0,0])
    # Plane: dot(plane_normal, P) + d = 0
    # → t = -d / dot(plane_normal, ray_dir)
    denom = rays @ plane_normal  # (N,)

    # Filter out nearly-parallel rays (no intersection)
    valid_mask = np.abs(denom) > 1e-9
    n_valid = valid_mask.sum()

    if n_valid == 0:
        logger.warning("triangulate: all rays parallel to laser plane — no intersection")
        return np.empty((0, 3), dtype=np.float64)

    t = np.full(n_points, np.nan, dtype=np.float64)
    t[valid_mask] = -d / denom[valid_mask]

    # 3D points in camera frame: (N, 3)
    points_cam = rays * t[:, np.newaxis]

    # Remove invalid points
    points_cam = points_cam[valid_mask]

    # ------------------------------------------------------------------ #
    # Step 4 — Rotate into world frame  (rotation around Y axis)
    # ------------------------------------------------------------------ #
    cos_a = np.cos(rotation_angle_rad)
    sin_a = np.sin(rotation_angle_rad)

    # Rotation matrix Ry(θ) — unrotates the turntable (inverse rotation)
    Ry = np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float64,
    )

    if axis_point is not None:
        # Rotate around the turntable axis (not the camera origin).
        # P_world = Ry @ (P_cam - axis_point)
        # World-frame origin is at the turntable centre.
        T = np.asarray(axis_point, dtype=np.float64).reshape(3)
        points_world = (Ry @ (points_cam - T).T).T
    else:
        points_world = (Ry @ points_cam.T).T  # (N, 3)

    logger.debug(
        "triangulate: %d input pixels → %d valid 3D points (angle=%.4f rad)",
        n_points,
        n_valid,
        rotation_angle_rad,
    )
    return points_world.astype(np.float64)
