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
    front_normal_z: float = -1.0,
    corner_shift: int = 0,
) -> np.ndarray | None:
    """Return known platform-frame ArUco marker corners on a rotating cube.

    At angle zero, side marker 0 faces platform -Z. With positive turntable
    angle, visible side markers follow 0 -> 1 -> 2 -> 3 -> 0 for a camera on
    the -Z side of the scanner.
    """
    side_ids = [int(value) for value in side_marker_ids]
    half_cube = float(cube_size_mm) / 2.0
    cube_center = np.asarray(cube_center_mm, dtype=np.float64).reshape(3)
    front_z = -1.0 if float(front_normal_z) < 0.0 else 1.0
    if front_z < 0.0:
        side_normals = (
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        )
    else:
        side_normals = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
    face_normals = {side_id: normal for side_id, normal in zip(side_ids, side_normals)}
    if top_marker_id is not None:
        face_normals[int(top_marker_id)] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = face_normals.get(int(marker_id))
    if normal is None:
        return None

    center_obj = cube_center + normal * half_cube
    corners_obj = _marker_corners_from_face(center_obj, normal, marker_size_mm)
    if int(corner_shift) % 4:
        corners_obj = np.roll(corners_obj, -int(corner_shift), axis=0)
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
    auto_layout: bool = True,
    angle_offset_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve camera-to-platform extrinsics from detected ArUco cube observations."""
    cv2, _aruco, _dictionary, _parameters = _get_aruco_api(dictionary_name)
    marker_counts: dict[int, int] = {}
    observed_marker_ids: set[int] = set()
    for observation in observations:
        for marker in observation.get("markers", []):
            marker_id = int(marker["id"])
            img = np.asarray(marker["corners"], dtype=np.float32).reshape(4, 2)
            marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
            if img.shape == (4, 2):
                observed_marker_ids.add(marker_id)

    used_observations = int(sum(marker_counts.values()))

    if used_observations < 3:
        from scanner.calibration import CalibrationError

        raise CalibrationError(
            f"At least 3 marker observations are required, got {used_observations}"
        )

    def _build_points(
        angle_sign: float,
        angle_offset_rad: float,
        front_normal_z: float,
        corner_shifts: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for observation in observations:
            angle = float(observation["angle_rad"]) * float(angle_sign) + float(angle_offset_rad)
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
                    front_normal_z=front_normal_z,
                    corner_shift=corner_shifts.get(marker_id, 0),
                )
                if obj is None:
                    continue
                img = np.asarray(marker["corners"], dtype=np.float32).reshape(4, 2)
                object_points.append(obj)
                image_points.append(img)
        if not object_points:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        return (
            np.concatenate(object_points, axis=0).astype(np.float32),
            np.concatenate(image_points, axis=0).astype(np.float32),
        )

    def _solve_candidate(
        angle_sign: float,
        angle_offset_rad: float,
        front_normal_z: float,
        corner_shifts: dict[int, int],
    ) -> dict | None:
        obj_all, img_all = _build_points(
            angle_sign,
            angle_offset_rad,
            front_normal_z,
            corner_shifts,
        )
        if obj_all.shape[0] < 12:
            return None
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
            return None
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
        projected, _ = cv2.projectPoints(obj_all, rvec, tvec, camera_matrix, dist)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - img_all, axis=1)
        if inliers is not None and int(len(inliers)) > 0:
            inlier_errors = errors[inliers.reshape(-1)]
            inlier_count = int(len(inliers))
        else:
            inlier_errors = errors
            inlier_count = int(obj_all.shape[0])
        return {
            "obj_all": obj_all,
            "img_all": img_all,
            "rvec": rvec,
            "tvec": tvec,
            "inliers": inliers,
            "inlier_count": inlier_count,
            "mean_inlier_error": float(np.mean(inlier_errors)),
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "angle_sign": float(angle_sign),
            "angle_offset_rad": float(angle_offset_rad),
            "front_normal_z": float(front_normal_z),
            "corner_shifts": dict(corner_shifts),
        }

    dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    ids_for_layout = [
        marker_id
        for marker_id in [*side_marker_ids, top_marker_id]
        if marker_id is not None and int(marker_id) in observed_marker_ids
    ]
    if auto_layout and angle_offset_deg is None:
        coarse_offsets = np.deg2rad(np.arange(0.0, 360.0, 5.0, dtype=np.float64))
        angle_signs = (1.0, -1.0)
        front_normals = (-1.0, 1.0)
    elif angle_offset_deg is None:
        coarse_offsets = np.array([0.0], dtype=np.float64)
        angle_signs = (1.0, -1.0)
        front_normals = (-1.0, 1.0)
    else:
        coarse_offsets = np.array([np.deg2rad(float(angle_offset_deg))], dtype=np.float64)
        angle_signs = (1.0,)
        front_normals = (-1.0,)

    best: dict | None = None
    candidates_tested = 0
    zero_corner_shifts = {int(marker_id): 0 for marker_id in ids_for_layout}

    def _try_candidate(
        angle_sign: float,
        angle_offset_rad: float,
        front_normal_z: float,
        corner_shifts: dict[int, int],
    ) -> None:
        nonlocal best, best_score, candidates_tested
        candidates_tested += 1
        candidate = _solve_candidate(
            angle_sign,
            angle_offset_rad,
            front_normal_z,
            corner_shifts,
        )
        if candidate is None:
            return
        candidate_score = (
            -candidate["inlier_count"],
            candidate["mean_inlier_error"],
            candidate["mean_error"],
        )
        if best is None:
            best = candidate
            best_score = candidate_score
        elif candidate_score < best_score:
            best = candidate
            best_score = candidate_score

    best_score = (0, float("inf"), float("inf"))
    for angle_offset_rad in coarse_offsets:
        for angle_sign in angle_signs:
            for front_normal_z in front_normals:
                _try_candidate(
                    angle_sign,
                    float(angle_offset_rad),
                    front_normal_z,
                    zero_corner_shifts,
                )

    if auto_layout and angle_offset_deg is None and best is not None:
        base_offset = float(best["angle_offset_rad"])
        fine_offsets = base_offset + np.deg2rad(np.arange(-5.0, 5.01, 0.5, dtype=np.float64))
        angle_sign = float(best["angle_sign"])
        front_normal_z = float(best["front_normal_z"])
        for angle_offset_rad in fine_offsets:
            _try_candidate(
                angle_sign,
                float(angle_offset_rad),
                front_normal_z,
                zero_corner_shifts,
            )

        # If the user mounted one or more tags rotated by 90-degree steps, try
        # single-marker corner shifts around the best angular convention. This
        # keeps runtime bounded while covering the common physical mistake.
        current_best_offset = float(best["angle_offset_rad"])
        current_best_sign = float(best["angle_sign"])
        current_best_front = float(best["front_normal_z"])
        for marker_id in ids_for_layout:
            for shift in (1, 2, 3):
                corner_shifts = dict(zero_corner_shifts)
                corner_shifts[int(marker_id)] = shift
                candidates_tested += 1
                candidate = _solve_candidate(
                    current_best_sign,
                    current_best_offset,
                    current_best_front,
                    corner_shifts,
                )
                if candidate is None:
                    continue
                candidate_score = (
                    -candidate["inlier_count"],
                    candidate["mean_inlier_error"],
                    candidate["mean_error"],
                )
                if best is None:
                    best = candidate
                    best_score = candidate_score
                elif candidate_score < best_score:
                    best = candidate
                    best_score = candidate_score

    if best is None:
        from scanner.calibration import CalibrationError

        raise CalibrationError("solvePnPRansac failed for all ArUco cube layout candidates")

    obj_all = best["obj_all"]
    img_all = best["img_all"]
    rvec = best["rvec"]
    tvec = best["tvec"]
    inliers = best["inliers"]

    world_to_camera_rotation, _ = cv2.Rodrigues(rvec)
    camera_to_platform_rotation = world_to_camera_rotation.T
    camera_to_platform_translation = (-camera_to_platform_rotation @ tvec.reshape(3)).reshape(3)

    projected, _ = cv2.projectPoints(obj_all, rvec, tvec, camera_matrix, dist)
    errors = np.linalg.norm(projected.reshape(-1, 2) - img_all, axis=1)
    inlier_count = int(best["inlier_count"])
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
        "mean_inlier_reprojection_error_px": float(best["mean_inlier_error"]),
        "max_reprojection_error_px": float(np.max(errors)),
        "markers": {str(key): int(value) for key, value in sorted(marker_counts.items())},
        "auto_layout": bool(auto_layout),
        "layout_candidates_tested": int(candidates_tested),
        "selected_angle_sign": float(best["angle_sign"]),
        "selected_angle_offset_deg": float(
            (np.rad2deg(float(best["angle_offset_rad"])) + 360.0) % 360.0
        ),
        "selected_front_normal_z": float(best["front_normal_z"]),
        "selected_corner_shifts": {
            str(key): int(value) for key, value in sorted(best["corner_shifts"].items())
        },
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
