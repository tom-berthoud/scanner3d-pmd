"""scanner.processing.triangulation - Laser triangulation pixel to 3D."""

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
    camera_to_platform_rotation: np.ndarray | None = None,
    camera_to_platform_translation: np.ndarray | None = None,
) -> np.ndarray:
    """Convert laser line pixels to 3D object-frame points.

    The laser plane is expressed in the stationary platform frame. Pixel rays
    are first back-projected in the camera frame, transformed into the platform
    frame, intersected with the shared laser plane, then unrotated by the
    turntable angle around the Y axis.
    """
    if line_pixels.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    if line_pixels.ndim != 2 or line_pixels.shape[1] != 2:
        raise ValueError(f"line_pixels must be (N, 2), got {line_pixels.shape}")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be (3, 3), got {camera_matrix.shape}")
    if laser_plane.shape != (4,):
        raise ValueError(f"laser_plane must be (4,), got {laser_plane.shape}")

    n_points = line_pixels.shape[0]
    dist = np.asarray(dist_coeffs, dtype=np.float64).flatten()
    if dist.size == 0 or np.allclose(dist, 0.0):
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        xy_norm = np.column_stack(
            [
                (line_pixels[:, 0].astype(np.float64) - cx) / fx,
                (line_pixels[:, 1].astype(np.float64) - cy) / fy,
            ]
        )
    else:
        import cv2  # type: ignore[import]

        pts_distorted = line_pixels.reshape(-1, 1, 2).astype(np.float64)
        pts_undistorted = cv2.undistortPoints(pts_distorted, camera_matrix, dist_coeffs)
        xy_norm = pts_undistorted.reshape(-1, 2)
    rays_cam = np.hstack([xy_norm, np.ones((n_points, 1), dtype=np.float64)])

    if camera_to_platform_rotation is None:
        rot = np.eye(3, dtype=np.float64)
    else:
        rot = np.asarray(camera_to_platform_rotation, dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError(f"camera_to_platform_rotation must be (3, 3), got {rot.shape}")

    if camera_to_platform_translation is None:
        trans = np.zeros(3, dtype=np.float64)
    else:
        trans = np.asarray(camera_to_platform_translation, dtype=np.float64).reshape(3)

    rays_platform = (rot @ rays_cam.T).T

    a, b, c, d = laser_plane.astype(np.float64)
    plane_normal = np.array([a, b, c], dtype=np.float64)
    denom = rays_platform @ plane_normal
    numer = -(float(trans @ plane_normal) + d)
    valid_mask = np.abs(denom) > 1e-9
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        logger.warning("triangulate: all rays parallel to laser plane")
        return np.empty((0, 3), dtype=np.float64)

    t = np.full(n_points, np.nan, dtype=np.float64)
    t[valid_mask] = numer / denom[valid_mask]
    points_platform = trans + rays_platform[valid_mask] * t[valid_mask, np.newaxis]

    cos_a = np.cos(rotation_angle_rad)
    sin_a = np.sin(rotation_angle_rad)
    ry = np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float64,
    )

    if axis_point is not None:
        axis = np.asarray(axis_point, dtype=np.float64).reshape(3)
        points_world = (ry @ (points_platform - axis).T).T
    else:
        points_world = (ry @ points_platform.T).T

    logger.debug(
        "triangulate: %d input pixels -> %d valid 3D points (angle=%.4f rad)",
        n_points,
        n_valid,
        rotation_angle_rad,
    )
    return points_world.astype(np.float64)
