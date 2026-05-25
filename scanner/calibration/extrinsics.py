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


def _checkerboard_object_point_variants(
    object_points: np.ndarray,
    board_size: tuple[int, int],
) -> list[tuple[str, np.ndarray]]:
    """Return plausible detector-to-board corner orderings for one checkerboard."""
    cols, rows = board_size
    grid = np.asarray(object_points, dtype=np.float32).reshape(rows, cols, 3)
    variants = [
        ("origin_col_row", grid),
        ("origin_last_col", grid[:, ::-1, :]),
        ("origin_last_row", grid[::-1, :, :]),
        ("origin_opposite_corner", grid[::-1, ::-1, :]),
    ]
    return [(name, variant.reshape(-1, 3).astype(np.float32)) for name, variant in variants]


def _axis_angle_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a right-handed rotation matrix around an arbitrary axis."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = axis / norm
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _rotate_points_around_axis(
    points: np.ndarray,
    angle_rad: float,
    axis: np.ndarray,
    axis_point: np.ndarray,
) -> np.ndarray:
    """Rotate platform-frame points around the turntable axis."""
    rot = _axis_angle_rotation_matrix(axis, angle_rad)
    origin = np.asarray(axis_point, dtype=np.float64).reshape(3)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return ((rot @ (pts - origin).T).T + origin).astype(np.float32)


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


def _camera_pose_from_pnp(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV platform-to-camera PnP pose into camera-to-platform pose."""
    import cv2  # type: ignore[import]

    rot_platform_to_cam, _ = cv2.Rodrigues(rvec)
    rot_camera_to_platform = rot_platform_to_cam.T
    trans_camera_to_platform = (-rot_camera_to_platform @ tvec.reshape(3)).astype(np.float64)
    return rot_camera_to_platform, trans_camera_to_platform


def _pnp_guess_from_camera_pose(
    rotation_camera_to_platform: np.ndarray,
    translation_camera_to_platform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-platform pose into OpenCV platform-to-camera rvec/tvec."""
    import cv2  # type: ignore[import]

    rot_camera_to_platform = np.asarray(rotation_camera_to_platform, dtype=np.float64).reshape(3, 3)
    trans_camera_to_platform = np.asarray(translation_camera_to_platform, dtype=np.float64).reshape(3)
    rot_platform_to_cam = rot_camera_to_platform.T
    trans_platform_to_cam = -rot_platform_to_cam @ trans_camera_to_platform
    rvec, _ = cv2.Rodrigues(rot_platform_to_cam)
    return rvec.astype(np.float64), trans_platform_to_cam.reshape(3, 1).astype(np.float64)


def _tvec_from_camera_position(rvec: np.ndarray, camera_position_platform: np.ndarray) -> np.ndarray:
    """Return OpenCV tvec for a fixed camera center in the platform frame."""
    import cv2  # type: ignore[import]

    rot_platform_to_cam, _ = cv2.Rodrigues(rvec)
    camera_position = np.asarray(camera_position_platform, dtype=np.float64).reshape(3)
    return (-rot_platform_to_cam @ camera_position).reshape(3, 1).astype(np.float64)


def _fixed_translation_rotation_fit(
    object_point_variants: list[tuple[str, np.ndarray]],
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    camera_position_platform: np.ndarray,
    initial_camera_to_platform_rotation: np.ndarray,
) -> dict:
    """Fit only camera orientation while keeping the measured camera position fixed."""
    import cv2  # type: ignore[import]
    from scipy.optimize import least_squares  # type: ignore[import]

    initial_rvec, _initial_tvec = _pnp_guess_from_camera_pose(
        initial_camera_to_platform_rotation,
        camera_position_platform,
    )
    image_points = image_points.reshape(-1, 2).astype(np.float64)

    def residuals(rvec_flat: np.ndarray, points: np.ndarray) -> np.ndarray:
        rvec = np.asarray(rvec_flat, dtype=np.float64).reshape(3, 1)
        tvec = _tvec_from_camera_position(rvec, camera_position_platform)
        projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
        return (projected.reshape(-1, 2) - image_points).reshape(-1)

    best: dict | None = None
    for order_name, points in object_point_variants:
        result = least_squares(
            residuals,
            initial_rvec.reshape(3),
            args=(points,),
            method="trf",
            max_nfev=200,
        )
        rvec = result.x.reshape(3, 1).astype(np.float64)
        tvec = _tvec_from_camera_position(rvec, camera_position_platform)
        report = _extrinsics_report(
            points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
        candidate = {
            "corner_order": order_name,
            "object_points": points,
            "rvec": rvec,
            "tvec": tvec,
            "mean_reprojection_error_px": report["mean_reprojection_error_px"],
            "max_reprojection_error_px": report["max_reprojection_error_px"],
            "optimizer_success": bool(result.success),
        }
        if best is None or candidate["mean_reprojection_error_px"] < best["mean_reprojection_error_px"]:
            best = candidate

    if best is None:
        from scanner.calibration import CalibrationError

        raise CalibrationError("Fixed-translation extrinsics rotation fit failed")
    return best


def _solve_extrinsics_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    initial_camera_to_platform_rotation: np.ndarray | None = None,
    initial_camera_to_platform_translation: np.ndarray | None = None,
    max_prior_distance_mm: float | None = None,
    max_translation_fallback_distance_mm: float | None = None,
    max_fixed_translation_reprojection_px: float | None = None,
    object_point_variants: list[tuple[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """Solve PnP and select the pose consistent with the known mechanical layout."""
    import cv2  # type: ignore[import]

    if object_point_variants is None:
        object_point_variants = [("provided_order", np.asarray(object_points, dtype=np.float32))]

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]] = []

    for order_name, ordered_object_points in object_point_variants:
        try:
            result = cv2.solvePnPGeneric(
                ordered_object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            ok = bool(result[0])
            if ok:
                for rvec, tvec in zip(result[1], result[2]):
                    candidates.append(
                        (
                            np.asarray(rvec, dtype=np.float64),
                            np.asarray(tvec, dtype=np.float64),
                            ordered_object_points,
                            "ippe",
                            order_name,
                        )
                    )
        except cv2.error:
            pass

        if (
            initial_camera_to_platform_rotation is not None
            and initial_camera_to_platform_translation is not None
        ):
            guess_rvec, guess_tvec = _pnp_guess_from_camera_pose(
                initial_camera_to_platform_rotation,
                initial_camera_to_platform_translation,
            )
            ok, rvec, tvec = cv2.solvePnP(
                ordered_object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                rvec=guess_rvec,
                tvec=guess_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                candidates.append(
                    (
                        rvec,
                        tvec,
                        ordered_object_points,
                        "iterative_with_mechanical_guess",
                        order_name,
                    )
                )

    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            from scanner.calibration import CalibrationError

            raise CalibrationError("OpenCV solvePnP failed for camera extrinsics")
        candidates.append((rvec, tvec, object_points, "iterative", "provided_order"))

    candidate_reports = []
    for idx, (rvec, tvec, ordered_object_points, method, order_name) in enumerate(candidates):
        report = _extrinsics_report(
            ordered_object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
        _rotation, position = _camera_pose_from_pnp(rvec, tvec)
        prior_distance = None
        if initial_camera_to_platform_translation is not None:
            prior = np.asarray(initial_camera_to_platform_translation, dtype=np.float64).reshape(3)
            prior_distance = float(np.linalg.norm(position - prior))
        candidate_reports.append(
            {
                "index": idx,
                "method": method,
                "corner_order": order_name,
                "rvec": rvec,
                "tvec": tvec,
                "object_points": ordered_object_points,
                "camera_position_mm": position,
                "mean_reprojection_error_px": report["mean_reprojection_error_px"],
                "max_reprojection_error_px": report["max_reprojection_error_px"],
                "prior_distance_mm": prior_distance,
            }
        )

    if initial_camera_to_platform_translation is None:
        selected = min(candidate_reports, key=lambda item: item["mean_reprojection_error_px"])
    else:
        best_error = min(item["mean_reprojection_error_px"] for item in candidate_reports)
        error_limit = max(best_error * 3.0, best_error + 2.0)
        plausible = [
            item for item in candidate_reports if item["mean_reprojection_error_px"] <= error_limit
        ]
        selected = min(
            plausible,
            key=lambda item: (
                float("inf")
                if item["prior_distance_mm"] is None
                else float(item["prior_distance_mm"]),
                item["mean_reprojection_error_px"],
            ),
        )

    public_reports = []
    for item in candidate_reports:
        public_reports.append(
            {
                "index": item["index"],
                "method": item["method"],
                "corner_order": item["corner_order"],
                "selected": item is selected,
                "camera_position_mm": item["camera_position_mm"].tolist(),
                "mean_reprojection_error_px": item["mean_reprojection_error_px"],
                "max_reprojection_error_px": item["max_reprojection_error_px"],
                "prior_distance_mm": item["prior_distance_mm"],
            }
        )

    selection = {
        "translation_source": "pnp",
        "prior_distance_mm": selected["prior_distance_mm"],
    }
    if max_prior_distance_mm is not None and selected["prior_distance_mm"] is not None:
        if float(selected["prior_distance_mm"]) > float(max_prior_distance_mm):
            if (
                max_translation_fallback_distance_mm is not None
                and initial_camera_to_platform_translation is not None
                and initial_camera_to_platform_rotation is not None
                and float(selected["prior_distance_mm"])
                <= float(max_translation_fallback_distance_mm)
            ):
                fixed_fit = _fixed_translation_rotation_fit(
                    object_point_variants,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    initial_camera_to_platform_translation,
                    initial_camera_to_platform_rotation,
                )
                max_error = (
                    5.0
                    if max_fixed_translation_reprojection_px is None
                    else float(max_fixed_translation_reprojection_px)
                )
                if float(fixed_fit["mean_reprojection_error_px"]) > max_error:
                    from scanner.calibration import CalibrationError

                    error = CalibrationError(
                        "Extrinsics fixed-position rotation fit has too much reprojection error: "
                        f"{fixed_fit['mean_reprojection_error_px']:.2f} px "
                        f"(limit {max_error:.2f} px). Check mechanical camera position and "
                        "checkerboard platform coordinates."
                    )
                    error.report = {  # type: ignore[attr-defined]
                        "pnp_candidates": public_reports,
                        "fixed_translation_candidate": {
                            "corner_order": fixed_fit["corner_order"],
                            "mean_reprojection_error_px": fixed_fit[
                                "mean_reprojection_error_px"
                            ],
                            "max_reprojection_error_px": fixed_fit["max_reprojection_error_px"],
                            "camera_position_mm": np.asarray(
                                initial_camera_to_platform_translation,
                                dtype=np.float64,
                            )
                            .reshape(3)
                            .tolist(),
                        },
                    }
                    raise error
                selection["translation_source"] = "mechanical_prior"
                selection["rotation_source"] = "fixed_translation_reprojection"
                selection["translation_reason"] = (
                    "PnP translation exceeded strict prior limit; rotation was refit with "
                    "the measured camera position held fixed"
                )
                return (
                    fixed_fit["rvec"],
                    fixed_fit["tvec"],
                    fixed_fit["object_points"],
                    public_reports,
                    selection,
                )
            from scanner.calibration import CalibrationError

            error = CalibrationError(
                "Extrinsics pose is inconsistent with the mechanical camera position: "
                f"best candidate is {selected['prior_distance_mm']:.1f} mm away "
                f"(limit {max_prior_distance_mm:.1f} mm). Check checkerboard origin, axes "
                "and platform frame convention."
            )
            error.report = {  # type: ignore[attr-defined]
                "pnp_candidates": public_reports,
                "selected_candidate": public_reports[selected["index"]],
            }
            raise error

    return selected["rvec"], selected["tvec"], selected["object_points"], public_reports, selection


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
    initial_camera_to_platform_rotation: np.ndarray | None = None,
    initial_camera_to_platform_translation: np.ndarray | None = None,
    max_prior_distance_mm: float | None = None,
    max_translation_fallback_distance_mm: float | None = None,
    max_fixed_translation_reprojection_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate camera-to-platform extrinsics from one known checkerboard pose.

    The checkerboard object points are expressed directly in the platform
    frame. OpenCV estimates platform-to-camera pose, which is inverted before
    saving because the scan pipeline needs camera-to-platform transforms.  If
    a rough mechanical camera pose is provided, it is used to disambiguate the
    planar checkerboard PnP solutions.
    """
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

    object_point_variants = _checkerboard_object_point_variants(object_points, board_size)
    rvec, tvec, selected_object_points, candidates, selection = _solve_extrinsics_pnp(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        initial_camera_to_platform_rotation,
        initial_camera_to_platform_translation,
        max_prior_distance_mm=max_prior_distance_mm,
        max_translation_fallback_distance_mm=max_translation_fallback_distance_mm,
        max_fixed_translation_reprojection_px=max_fixed_translation_reprojection_px,
        object_point_variants=object_point_variants,
    )

    rot_camera_to_platform, trans_camera_to_platform = _camera_pose_from_pnp(rvec, tvec)
    if selection.get("translation_source") == "mechanical_prior":
        trans_camera_to_platform = np.asarray(
            initial_camera_to_platform_translation,
            dtype=np.float64,
        ).reshape(3)
    report = _extrinsics_report(
        selected_object_points,
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
    report["pnp_candidates"] = candidates
    report["translation_source"] = selection["translation_source"]
    report["rotation_source"] = selection.get("rotation_source", "pnp")
    report["prior_distance_mm"] = selection["prior_distance_mm"]
    if "translation_reason" in selection:
        report["translation_reason"] = selection["translation_reason"]

    if output_path:
        save_camera_extrinsics(
            rot_camera_to_platform,
            trans_camera_to_platform,
            output_path,
            report=report,
        )
    return rot_camera_to_platform, trans_camera_to_platform, report


def calibrate_camera_extrinsics_turntable(
    observations: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_size: tuple[int, int],
    square_size_mm: float,
    board_origin_mm: list[float] | tuple[float, float, float] | np.ndarray,
    board_col_axis: list[float] | tuple[float, float, float] | np.ndarray,
    board_row_axis: list[float] | tuple[float, float, float] | np.ndarray,
    rotation_axis: list[float] | tuple[float, float, float] | np.ndarray = (0.0, 1.0, 0.0),
    rotation_axis_point_mm: list[float] | tuple[float, float, float] | np.ndarray = (
        0.0,
        0.0,
        0.0,
    ),
    output_path: Optional[str] = None,
    initial_camera_to_platform_rotation: np.ndarray | None = None,
    initial_camera_to_platform_translation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Calibrate camera extrinsics from a checkerboard fixed on the rotating turntable.

    Each observation must contain ``image`` and ``angle_rad``. The checkerboard
    coordinates are specified at angle zero, then rotated around the platform
    turntable axis for each capture.
    """
    import cv2  # type: ignore[import]
    from scipy.optimize import least_squares  # type: ignore[import]

    from scanner.calibration import CalibrationError

    if len(observations) < 3:
        raise CalibrationError(f"At least 3 turntable observations are required, got {len(observations)}")

    base_points = _platform_board_points(
        board_size,
        square_size_mm,
        np.asarray(board_origin_mm, dtype=np.float64),
        np.asarray(board_col_axis, dtype=np.float64),
        np.asarray(board_row_axis, dtype=np.float64),
    )
    axis = np.asarray(rotation_axis, dtype=np.float64).reshape(3)
    axis_point = np.asarray(rotation_axis_point_mm, dtype=np.float64).reshape(3)

    detected = []
    for idx, observation in enumerate(observations):
        image = observation.get("image")
        angle_rad = float(observation.get("angle_rad", 0.0))
        found, corners, _gray = _detect_checkerboard(image, board_size)
        if not found or corners is None:
            raise CalibrationError(f"Checkerboard not found in turntable observation {idx + 1}")
        image_points = corners.reshape(-1, 2).astype(np.float32)
        detected.append(
            {
                "index": idx,
                "angle_rad": angle_rad,
                "image_points": image_points,
            }
        )

    def select_points_for_sign(angle_sign: float) -> tuple[list[dict], list[dict], float]:
        selected_observations = []
        selected_reports = []
        total_error = 0.0
        for item in detected:
            rotated_points = _rotate_points_around_axis(
                base_points,
                angle_sign * float(item["angle_rad"]),
                axis,
                axis_point,
            )
            variants = _checkerboard_object_point_variants(rotated_points, board_size)
            rvec, tvec, selected_points, candidates, _selection = _solve_extrinsics_pnp(
                rotated_points,
                item["image_points"],
                camera_matrix,
                dist_coeffs,
                initial_camera_to_platform_rotation=initial_camera_to_platform_rotation,
                initial_camera_to_platform_translation=initial_camera_to_platform_translation,
                object_point_variants=variants,
            )
            one_report = _extrinsics_report(
                selected_points,
                item["image_points"],
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
            total_error += float(one_report["mean_reprojection_error_px"])
            selected_observations.append(
                {
                    "index": item["index"],
                    "angle_rad": item["angle_rad"],
                    "object_points": selected_points,
                    "image_points": item["image_points"],
                }
            )
            selected_candidate = next(
                (candidate for candidate in candidates if candidate.get("selected")),
                candidates[0],
            )
            selected_reports.append(
                {
                    "index": item["index"],
                    "angle_rad": item["angle_rad"],
                    "corner_order": selected_candidate.get("corner_order"),
                    "single_view_mean_reprojection_error_px": one_report[
                        "mean_reprojection_error_px"
                    ],
                    "single_view_camera_position_mm": selected_candidate.get("camera_position_mm"),
                    "single_view_prior_distance_mm": selected_candidate.get("prior_distance_mm"),
                }
            )
        return selected_observations, selected_reports, total_error / max(len(detected), 1)

    candidates_by_sign = []
    for angle_sign in (1.0, -1.0):
        selected_observations, selected_reports, mean_error = select_points_for_sign(angle_sign)
        candidates_by_sign.append((mean_error, angle_sign, selected_observations, selected_reports))
    _mean_error, angle_sign, selected_observations, selected_reports = min(
        candidates_by_sign,
        key=lambda item: item[0],
    )

    if (
        initial_camera_to_platform_rotation is not None
        and initial_camera_to_platform_translation is not None
    ):
        initial_rvec, initial_tvec = _pnp_guess_from_camera_pose(
            initial_camera_to_platform_rotation,
            initial_camera_to_platform_translation,
        )
    else:
        first = selected_observations[0]
        ok, initial_rvec, initial_tvec = cv2.solvePnP(
            first["object_points"],
            first["image_points"],
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise CalibrationError("OpenCV solvePnP failed for initial turntable extrinsics")

    def residuals(params: np.ndarray) -> np.ndarray:
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)
        residual_chunks = []
        for item in selected_observations:
            projected, _ = cv2.projectPoints(
                item["object_points"],
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            residual_chunks.append((projected.reshape(-1, 2) - item["image_points"]).reshape(-1))
        return np.concatenate(residual_chunks)

    x0 = np.concatenate([initial_rvec.reshape(3), initial_tvec.reshape(3)])
    result = least_squares(residuals, x0, method="trf", max_nfev=400)
    if not result.success:
        raise CalibrationError(f"Turntable extrinsics optimisation failed: {result.message}")

    rvec = result.x[:3].reshape(3, 1).astype(np.float64)
    tvec = result.x[3:6].reshape(3, 1).astype(np.float64)
    rotation, translation = _camera_pose_from_pnp(rvec, tvec)

    per_capture = []
    all_errors = []
    for item, selected_report in zip(selected_observations, selected_reports):
        report = _extrinsics_report(
            item["object_points"],
            item["image_points"],
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
        per_capture.append({**selected_report, **report})
        projected, _ = cv2.projectPoints(
            item["object_points"],
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        errors = np.linalg.norm(projected.reshape(-1, 2) - item["image_points"], axis=1)
        all_errors.extend(float(value) for value in errors)

    all_errors_arr = np.asarray(all_errors, dtype=np.float64)
    report = {
        "method": "turntable_checkerboard",
        "captures": len(selected_observations),
        "points": int(sum(len(item["object_points"]) for item in selected_observations)),
        "angle_sign": angle_sign,
        "mean_reprojection_error_px": float(all_errors_arr.mean()),
        "max_reprojection_error_px": float(all_errors_arr.max()),
        "camera_position_mm": translation.tolist(),
        "board_origin_mm": [float(v) for v in board_origin_mm],
        "board_col_axis": [float(v) for v in board_col_axis],
        "board_row_axis": [float(v) for v in board_row_axis],
        "rotation_axis": axis.tolist(),
        "rotation_axis_point_mm": axis_point.tolist(),
        "per_capture": per_capture,
    }
    if initial_camera_to_platform_translation is not None:
        prior = np.asarray(initial_camera_to_platform_translation, dtype=np.float64).reshape(3)
        report["prior_distance_mm"] = float(np.linalg.norm(translation - prior))

    if output_path:
        save_camera_extrinsics(rotation, translation, output_path, report=report)
    return rotation, translation, report


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
