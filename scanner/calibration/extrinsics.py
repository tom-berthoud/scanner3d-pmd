"""Utilities for camera extrinsics calibration files."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _normalize(vec: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError(f"{name} vector is degenerate")
    return vec / norm


def _rotation_y(angle_rad: float) -> np.ndarray:
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float64,
    )


def _marker_corners_from_face(
    center: np.ndarray,
    normal: np.ndarray,
    marker_size_mm: float,
) -> np.ndarray:
    """Return object corners matching OpenCV ArUco corner order.

    The marker is assumed readable from outside the cube with its top toward
    platform +Y. Corner order is top-left, top-right, bottom-right, bottom-left.
    """
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = _normalize(normal, "face normal")
    if abs(float(normal @ up)) > 0.95:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        down = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        view_direction = -normal
        right = _normalize(np.cross(up, view_direction), "marker right")
        down = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    half = float(marker_size_mm) / 2.0
    return np.array(
        [
            center - right * half - down * half,
            center + right * half - down * half,
            center + right * half + down * half,
            center - right * half + down * half,
        ],
        dtype=np.float32,
    )


def aruco_cube_marker_points(
    marker_id: int,
    rotation_angle_rad: float,
    cube_size_mm: float = 30.0,
    marker_size_mm: float = 20.0,
    cube_center_mm: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 15.0, 0.0),
    side_marker_ids: list[int] | tuple[int, int, int, int] = (0, 1, 2, 3),
    top_marker_id: int | None = 4,
) -> np.ndarray | None:
    """Return known platform-frame ArUco marker corners on a rotating cube.

    At angle zero, side marker 0 faces platform -Z. With positive turntable
    angle, visible side markers follow 0 -> 1 -> 2 -> 3 -> 0 for a camera on
    the -Z side of the scanner.
    """
    side_ids = [int(value) for value in side_marker_ids]
    half_cube = float(cube_size_mm) / 2.0
    cube_center = np.asarray(cube_center_mm, dtype=np.float64).reshape(3)
    face_normals = {
        side_ids[0]: np.array([0.0, 0.0, -1.0], dtype=np.float64),
        side_ids[1]: np.array([1.0, 0.0, 0.0], dtype=np.float64),
        side_ids[2]: np.array([0.0, 0.0, 1.0], dtype=np.float64),
        side_ids[3]: np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    }
    if top_marker_id is not None:
        face_normals[int(top_marker_id)] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = face_normals.get(int(marker_id))
    if normal is None:
        return None

    center_obj = cube_center + normal * half_cube
    corners_obj = _marker_corners_from_face(center_obj, normal, marker_size_mm)
    rot = _rotation_y(float(rotation_angle_rad))
    return (rot @ corners_obj.T).T.astype(np.float32)


def _get_aruco_api(dictionary_name: str):
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        from scanner.calibration import CalibrationError

        raise CalibrationError(
            "OpenCV is required for ArUco extrinsics calibration. "
            "Install opencv-contrib-python, not only opencv-python."
        ) from exc
    if not hasattr(cv2, "aruco"):
        from scanner.calibration import CalibrationError

        raise CalibrationError(
            "This OpenCV build has no cv2.aruco module. Install opencv-contrib-python."
        )

    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary_name, None)
    if dict_id is None:
        from scanner.calibration import CalibrationError

        raise CalibrationError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()
    return cv2, aruco, dictionary, parameters


def detect_aruco_markers(
    frame: np.ndarray,
    dictionary_name: str = "DICT_4X4_50",
) -> tuple[list[np.ndarray], list[int]]:
    """Detect ArUco markers and return corners plus integer IDs."""
    cv2, aruco, dictionary, parameters = _get_aruco_api(dictionary_name)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        corners, ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is None:
        return [], []
    return [np.asarray(c, dtype=np.float32).reshape(4, 2) for c in corners], [
        int(value) for value in ids.flatten()
    ]


def draw_aruco_overlay(
    frame: np.ndarray,
    corners: list[np.ndarray],
    ids: list[int],
) -> np.ndarray:
    """Draw detected marker outlines and IDs on a copy of *frame*."""
    cv2, _aruco, _dictionary, _parameters = _get_aruco_api("DICT_4X4_50")
    overlay = frame.copy()
    for marker_corners, marker_id in zip(corners, ids):
        pts = np.round(marker_corners).astype(int)
        cv2.polylines(overlay, [pts], True, (0, 255, 255), 2)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(
            overlay,
            str(marker_id),
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def solve_camera_extrinsics_from_aruco_cube(
    observations: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    cube_size_mm: float = 30.0,
    marker_size_mm: float = 20.0,
    cube_center_mm: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 15.0, 0.0),
    dictionary_name: str = "DICT_4X4_50",
    side_marker_ids: list[int] | tuple[int, int, int, int] = (0, 1, 2, 3),
    top_marker_id: int | None = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve camera-to-platform extrinsics from detected ArUco cube observations."""
    cv2, _aruco, _dictionary, _parameters = _get_aruco_api(dictionary_name)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    marker_counts: dict[int, int] = {}
    used_observations = 0

    for observation in observations:
        angle = float(observation["angle_rad"])
        for marker in observation.get("markers", []):
            marker_id = int(marker["id"])
            obj = aruco_cube_marker_points(
                marker_id,
                angle,
                cube_size_mm=cube_size_mm,
                marker_size_mm=marker_size_mm,
                cube_center_mm=cube_center_mm,
                side_marker_ids=side_marker_ids,
                top_marker_id=top_marker_id,
            )
            if obj is None:
                continue
            img = np.asarray(marker["corners"], dtype=np.float32).reshape(4, 2)
            object_points.append(obj)
            image_points.append(img)
            marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
            used_observations += 1

    if used_observations < 3:
        from scanner.calibration import CalibrationError

        raise CalibrationError(
            f"At least 3 marker observations are required, got {used_observations}"
        )

    obj_all = np.concatenate(object_points, axis=0).astype(np.float32)
    img_all = np.concatenate(image_points, axis=0).astype(np.float32)
    if obj_all.shape[0] < 12:
        from scanner.calibration import CalibrationError

        raise CalibrationError(f"At least 12 marker corners are required, got {obj_all.shape[0]}")

    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_all,
        img_all,
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,
        iterationsCount=200,
        confidence=0.995,
    )
    if not ok:
        from scanner.calibration import CalibrationError

        raise CalibrationError("solvePnPRansac failed for ArUco cube extrinsics")
    if inliers is not None and int(len(inliers)) >= 6:
        inlier_idx = inliers.reshape(-1)
        cv2.solvePnP(
            obj_all[inlier_idx],
            img_all[inlier_idx],
            camera_matrix,
            dist,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    world_to_camera_rotation, _ = cv2.Rodrigues(rvec)
    camera_to_platform_rotation = world_to_camera_rotation.T
    camera_to_platform_translation = (-camera_to_platform_rotation @ tvec.reshape(3)).reshape(3)

    projected, _ = cv2.projectPoints(obj_all, rvec, tvec, camera_matrix, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img_all, axis=1)
    inlier_count = int(len(inliers)) if inliers is not None else int(obj_all.shape[0])
    report = {
        "method": "aruco_cube_turntable",
        "dictionary": dictionary_name,
        "cube_size_mm": float(cube_size_mm),
        "marker_size_mm": float(marker_size_mm),
        "cube_center_mm": np.asarray(cube_center_mm, dtype=np.float64).reshape(3).tolist(),
        "side_marker_ids": [int(value) for value in side_marker_ids],
        "top_marker_id": None if top_marker_id is None else int(top_marker_id),
        "marker_observations": int(used_observations),
        "points": int(obj_all.shape[0]),
        "inliers": inlier_count,
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "markers": {str(key): int(value) for key, value in sorted(marker_counts.items())},
        "camera_position_mm": camera_to_platform_translation.tolist(),
    }
    logger.info(
        "ArUco cube extrinsics solved: observations=%d mean_error=%.3f px",
        used_observations,
        report["mean_reprojection_error_px"],
    )
    return camera_to_platform_rotation, camera_to_platform_translation, report


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
