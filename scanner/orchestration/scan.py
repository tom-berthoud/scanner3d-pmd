"""scanner.orchestration.scan — Main scan loop.

Orchestrates the full acquisition → processing → reconstruction → export
pipeline, managing the state machine throughout.
"""

import logging
import math
import time
from typing import Callable, Optional

import numpy as np

from scanner.orchestration.state_machine import ScannerState, StateMachine

logger = logging.getLogger(__name__)


def run_scan(
    config: dict,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    artifact_callback: Optional[Callable[[dict], None]] = None,
    state_machine: Optional[StateMachine] = None,
) -> str:
    """Run a full 3D scan and return the path to the exported file.

    Orchestration pipeline:
        1. Transition → SCANNING
        2. Run capture sequence (motor + laser + camera per step)
        3. Transition → PROCESSING
        4. For each frame: extract laser line + triangulate to 3D profile
        5. Merge profiles → point cloud; filter outliers
        6. Transition → EXPORTING
        7. Export raw cloud + STL or OBJ
        8. Transition → COMPLETE

    On any error, transitions to ERROR and re-raises the exception.

    Args:
        config: Full settings dict loaded from settings.yaml.
        progress_callback: Optional callable(current, total, message).
            Called at key milestones with current step, total steps, and
            a human-readable message.
        state_machine: Optional existing StateMachine instance.  If None,
            a fresh one is created.

    Returns:
        Absolute path to the exported STL or OBJ file.

    Raises:
        HardwareError: if a hardware operation fails.
        CalibrationError: if calibration files are missing or corrupt.
        RuntimeError: for any other scan pipeline failure.
    """
    from scanner.hardware import (
        HardwareError,
        camera_capture_all,
        camera_set_exposure,
        check_door_interlock,
        laser_set,
        motor_step,
    )
    from scanner.calibration import (
        CalibrationError,
        camera_ids,
        load_camera_model,
    )
    from scanner.calibration.multi_camera import _load_extrinsics
    from scanner.acquisition import run_capture_sequence_multi
    from scanner.processing import extract_laser_line, triangulate
    from scanner.reconstruction import (
        add_flat_caps_aligned,
        detect_top_plane,
        filter_outliers,
        fuse_half_turn_profiles,
        merge_profiles,
    )
    from scanner.export import export_stl, export_obj, export_point_cloud_ply

    sm = state_machine or StateMachine()

    # Load platform calibration (rotation axis point) if available
    import os
    import yaml as _yaml
    import numpy as _np

    _platform_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "platform.yaml")
    _axis_point = None
    if os.path.exists(_platform_path):
        try:
            _plat = _yaml.safe_load(open(_platform_path))
            _ap = _plat.get("rotation_axis_point_mm")
            if _ap is not None:
                _axis_point = _np.array(_ap, dtype=float)
        except Exception:
            pass

    scan_cfg = config.get("scan", {})
    n_steps: int = int(scan_cfg.get("n_steps", 200))
    exposure_calib_cfg = scan_cfg.get("exposure_calibration", {}) or {}

    proc_cfg = config.get("processing", {})
    default_threshold: int = int(proc_cfg.get("laser_threshold", 180))
    min_pixels: int = int(proc_cfg.get("min_line_pixels", 10))
    subpixel: bool = bool(proc_cfg.get("subpixel", True))
    extraction_mode: str = str(proc_cfg.get("extraction_mode", "row_mean"))

    recon_cfg = config.get("reconstruction", {})
    nb_neighbors: int = int(recon_cfg.get("outlier_nb_neighbors", 20))
    std_ratio: float = float(recon_cfg.get("outlier_std_ratio", 2.0))
    flat_caps_cfg = recon_cfg.get("flat_caps", {}) or {}
    top_plane_clip_cfg = recon_cfg.get("top_plane_clip", {}) or {}
    auto_cheat_cfg = recon_cfg.get("auto_cheat_extrinsics", {}) or {}
    profile_fusion_cfg = recon_cfg.get("profile_fusion", {}) or {}
    half_turn_fusion_cfg = recon_cfg.get("half_turn_profile_fusion", {}) or {}

    export_cfg = config.get("export", {})
    fmt: str = export_cfg.get("default_format", "stl").lower()
    output_dir: str = export_cfg.get("output_dir", "/tmp/scans")
    poisson_cfg = export_cfg.get("poisson", {})
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"scan_{timestamp}.{fmt}")
    cloud_path = os.path.join(output_dir, f"scan_{timestamp}_cloud.ply")

    def _artifact(kind: str, path: str, label: str, media_type: str, points: int | None = None) -> None:
        if artifact_callback is None:
            return
        payload = {
            "kind": kind,
            "path": os.path.abspath(path),
            "label": label,
            "media_type": media_type,
            "available": os.path.exists(path),
        }
        if points is not None:
            payload["points"] = int(points)
        artifact_callback(payload)

    def _progress(current: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(current, total, message)

    def _safe_laser_off() -> None:
        """Emergency laser shutdown — never raise."""
        try:
            laser_set(False)
        except Exception as exc:
            logger.error("Emergency laser off failed: %s", exc)

    def _go_error(exc: Exception) -> None:
        """Transition to ERROR state and ensure laser is off."""
        _safe_laser_off()
        try:
            sm.transition(ScannerState.ERROR)
        except ValueError:
            logger.warning("Could not transition to ERROR (current=%s)", sm.current_state.name)
        logger.error("Scan failed: %s", exc)

    def _sampling_config(cam_cfg: dict) -> dict[str, int]:
        sampling = cam_cfg.get("laser_sampling", {}) or {}
        return {
            "x_stride": max(1, int(sampling.get("x_stride", 1))),
            "y_stride": max(1, int(sampling.get("y_stride", 1))),
            "x_offset": max(0, int(sampling.get("x_offset", 0))),
            "y_offset": max(0, int(sampling.get("y_offset", 0))),
        }

    def _find_cam_cfg(camera_id: str) -> dict:
        return next(
            (item for item in config.get("cameras", []) if str(item.get("id")) == str(camera_id)),
            {},
        )

    def _measure_laser_width_px(frame: np.ndarray, camera_id: str, threshold: int) -> float | None:
        from scanner.processing.laser_line import _mask_shapes

        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        cam_cfg = _find_cam_cfg(camera_id)
        green = frame[:, :, 1]
        active = green >= int(threshold)
        active = _mask_shapes(
            active,
            cam_cfg.get("laser_mask", []) or [],
            original_shape=frame.shape[:2],
        )
        widths = active.sum(axis=1)
        widths = widths[widths > 0]
        if widths.size == 0:
            return None
        return float(np.median(widths))

    def _run_pre_scan_exposure_calibration() -> None:
        if not bool(exposure_calib_cfg.get("enabled", False)):
            return

        target_width = float(exposure_calib_cfg.get("target_median_width_px", 6.0))
        if target_width <= 0:
            logger.warning("exposure_calibration: invalid target width %.3f, skip", target_width)
            return

        sample_count = max(1, int(exposure_calib_cfg.get("sample_count", 3)))
        angle_step_deg = float(exposure_calib_cfg.get("angle_step_deg", 30.0))
        tolerance_width = max(0.0, float(exposure_calib_cfg.get("tolerance_width_px", 1.0)))
        max_iterations = max(1, int(exposure_calib_cfg.get("max_iterations", 4)))
        min_exposure = int(exposure_calib_cfg.get("min_exposure_us", 200))
        max_exposure = int(exposure_calib_cfg.get("max_exposure_us", 60000))
        max_adjust_factor = max(1.0, float(exposure_calib_cfg.get("max_adjust_factor", 3.0)))

        motor_cfg = config.get("motor", {})
        direction = str(scan_cfg.get("direction", "clockwise"))
        total_motor_steps = int(motor_cfg.get("steps_per_rev", 200)) * int(
            motor_cfg.get("microstepping", 1)
        )
        steps_per_sample = max(1, int(round(total_motor_steps * angle_step_deg / 360.0)))

        _progress(0, n_steps, "Calibrating camera exposure")

        def _capture_width_cycle() -> dict[str, float | None]:
            widths_by_camera: dict[str, list[float]] = {}
            seen_cameras: set[str] = set()
            try:
                for sample_idx in range(sample_count):
                    check_door_interlock()
                    laser_set(True)
                    check_door_interlock()
                    frames = camera_capture_all()
                    laser_set(False)

                    for camera_id, frame in frames.items():
                        seen_cameras.add(camera_id)
                        cam_cfg = _find_cam_cfg(camera_id)
                        threshold = int(cam_cfg.get("laser_threshold", default_threshold))
                        width = _measure_laser_width_px(frame, camera_id, threshold)
                        if width is not None:
                            widths_by_camera.setdefault(camera_id, []).append(width)

                    if sample_idx < sample_count - 1:
                        motor_step(steps_per_sample, direction)
            finally:
                _safe_laser_off()

            measured_by_camera: dict[str, float | None] = {}
            for camera_id in seen_cameras:
                widths = widths_by_camera.get(camera_id, [])
                measured_by_camera[camera_id] = (
                    float(np.median(np.asarray(widths, dtype=np.float64))) if widths else None
                )
            return measured_by_camera

        for iteration in range(1, max_iterations + 1):
            measured_by_camera = _capture_width_cycle()
            if not measured_by_camera:
                logger.warning("exposure_calibration: no camera frames captured, skip")
                return

            all_in_tolerance = True
            for camera_id, measured in measured_by_camera.items():
                cam_cfg = _find_cam_cfg(camera_id)
                old_exposure = int(
                    cam_cfg.get("exposure_us", config.get("camera", {}).get("exposure_us", 1000))
                )

                if measured is None or measured <= 0:
                    all_in_tolerance = False
                    factor = max_adjust_factor
                    measured_text = "none"
                else:
                    error = abs(measured - target_width)
                    if error <= tolerance_width:
                        logger.info(
                            "exposure_calibration: iter=%d camera=%s width=%.2fpx target=%.2fpx "
                            "error=%.2fpx exposure=%dus ok",
                            iteration,
                            camera_id,
                            measured,
                            target_width,
                            error,
                            old_exposure,
                        )
                        continue
                    all_in_tolerance = False
                    factor = target_width / measured
                    measured_text = f"{measured:.2f}"

                factor = max(1.0 / max_adjust_factor, min(max_adjust_factor, factor))
                new_exposure = int(round(old_exposure * factor))
                new_exposure = max(min_exposure, min(max_exposure, new_exposure))
                if new_exposure == old_exposure:
                    logger.info(
                        "exposure_calibration: iter=%d camera=%s width=%s target=%.2fpx "
                        "exposure unchanged=%dus",
                        iteration,
                        camera_id,
                        measured_text,
                        target_width,
                        old_exposure,
                    )
                    continue

                camera_set_exposure(new_exposure, gain=None, camera_id=camera_id)
                cam_cfg["exposure_us"] = new_exposure
                logger.info(
                    "exposure_calibration: iter=%d camera=%s width=%s target=%.2fpx "
                    "exposure %d -> %dus",
                    iteration,
                    camera_id,
                    measured_text,
                    target_width,
                    old_exposure,
                    new_exposure,
                )

            if all_in_tolerance:
                logger.info("exposure_calibration: converged in %d iteration(s)", iteration)
                return

        logger.info("exposure_calibration: reached max_iterations=%d", max_iterations)

    def _seed_vec_from_cfg(camera_id: str) -> np.ndarray | None:
        cam_cfg = _find_cam_cfg(camera_id)
        extr = cam_cfg.get("extrinsics", {}) or {}
        pos = extr.get("position_mm")
        yaw = extr.get("angle_camera_laser_deg", extr.get("yaw_deg"))
        elev = extr.get("angle_planxz_camera_deg", extr.get("elevation_deg", extr.get("pitch_deg")))
        if pos is None or yaw is None or elev is None:
            return None
        return np.asarray([float(pos[0]), float(pos[1]), float(pos[2]), float(yaw), float(elev)], dtype=np.float64)

    def _pose_from_vec(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot, trans = _load_extrinsics(
            {
                "extrinsics": {
                    "position_mm": [float(vec[0]), float(vec[1]), float(vec[2])],
                    "angle_camera_laser_deg": float(vec[3]),
                    "angle_planxz_camera_deg": float(vec[4]),
                    "up_mm": [0.0, 1.0, 0.0],
                }
            }
        )
        return rot, trans

    def _decimate_points(points: np.ndarray, max_points: int) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        return points[idx]

    def _build_cloud_from_extracted(
        extracted_items: list[dict],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        laser_plane: np.ndarray,
        cam_rot: np.ndarray,
        cam_trans: np.ndarray,
        max_points: int,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        per_line_limit = int(auto_cheat_cfg.get("per_line_max_points", 250))
        for item in extracted_items:
            line_px = item["line_pixels"]
            if line_px.shape[0] == 0:
                continue
            if line_px.shape[0] > per_line_limit:
                idx = np.linspace(0, line_px.shape[0] - 1, per_line_limit).astype(np.int64)
                line_px = line_px[idx]
            pts = triangulate(
                line_px,
                camera_matrix,
                dist_coeffs,
                laser_plane,
                float(item["angle_rad"]),
                axis_point=_axis_point,
                camera_to_platform_rotation=cam_rot,
                camera_to_platform_translation=cam_trans,
            )
            if pts.shape[0] > 0:
                parts.append(pts)
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return _decimate_points(np.vstack(parts).astype(np.float64), max_points=max_points)

    def _run_auto_cheat_extrinsics(
        extracted_by_camera: dict[str, list[dict]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not bool(auto_cheat_cfg.get("enabled", False)):
            return {}
        from scipy.optimize import minimize  # type: ignore[import]
        from scipy.spatial import cKDTree  # type: ignore[import]

        cams = [cid for cid in configured_camera_ids if cid in extracted_by_camera]
        if len(cams) < 2:
            return {}
        cam_a, cam_b = cams[0], cams[1]
        seed_a = _seed_vec_from_cfg(cam_a)
        seed_b = _seed_vec_from_cfg(cam_b)
        if seed_a is None or seed_b is None:
            logger.warning("auto_cheat_extrinsics: missing angle extrinsics seed in settings, skip")
            return {}

        cm_a, dc_a, lp_a, _, _ = camera_models[cam_a]
        cm_b, dc_b, lp_b, _, _ = camera_models[cam_b]

        x0 = np.hstack([seed_a, seed_b]).astype(np.float64)
        trans_span = float(auto_cheat_cfg.get("translation_span_mm", 120.0))
        angle_span = float(auto_cheat_cfg.get("angle_span_deg", 20.0))
        bounds = []
        for base in (seed_a, seed_b):
            bounds.extend(
                [
                    (base[0] - trans_span, base[0] + trans_span),
                    (base[1] - trans_span, base[1] + trans_span),
                    (base[2] - trans_span, base[2] + trans_span),
                    (base[3] - angle_span, base[3] + angle_span),
                    (base[4] - angle_span, base[4] + angle_span),
                ]
            )

        max_points = int(auto_cheat_cfg.get("max_points", 12000))
        best_score = float("inf")
        best_vec: np.ndarray | None = None

        def _score(vec: np.ndarray) -> float:
            nonlocal best_score, best_vec
            for i, (low, high) in enumerate(bounds):
                if vec[i] < low or vec[i] > high:
                    return 1e9
            rot_a, trans_a = _pose_from_vec(vec[:5])
            rot_b, trans_b = _pose_from_vec(vec[5:])
            cloud_a = _build_cloud_from_extracted(
                extracted_by_camera[cam_a], cm_a, dc_a, lp_a, rot_a, trans_a, max_points=max_points
            )
            cloud_b = _build_cloud_from_extracted(
                extracted_by_camera[cam_b], cm_b, dc_b, lp_b, rot_b, trans_b, max_points=max_points
            )
            if cloud_a.shape[0] < 200 or cloud_b.shape[0] < 200:
                return 1e9
            tree_a = cKDTree(cloud_a)
            tree_b = cKDTree(cloud_b)
            d_ba, _ = tree_a.query(cloud_b, k=1, workers=-1)
            d_ab, _ = tree_b.query(cloud_a, k=1, workers=-1)
            rms = 0.5 * (
                float(np.sqrt(np.mean(np.square(np.clip(d_ba, 0.0, 200.0)))))
                + float(np.sqrt(np.mean(np.square(np.clip(d_ab, 0.0, 200.0)))))
            )
            if rms < best_score:
                best_score = rms
                best_vec = vec.copy()
            return rms

        initial = _score(x0)
        res = minimize(
            _score,
            x0,
            method="Powell",
            options={
                "maxiter": int(auto_cheat_cfg.get("max_iter", 60)),
                "xtol": 1e-2,
                "ftol": 1e-2,
                "disp": False,
            },
        )
        final_vec = best_vec if best_vec is not None else np.asarray(res.x, dtype=np.float64)
        final = _score(final_vec)
        logger.info(
            "auto_cheat_extrinsics: cams=(%s,%s) initial=%.3fmm final=%.3fmm iters=%s success=%s",
            cam_a,
            cam_b,
            float(initial),
            float(final),
            int(getattr(res, "nit", 0)),
            bool(getattr(res, "success", False)),
        )
        rot_a, trans_a = _pose_from_vec(final_vec[:5])
        rot_b, trans_b = _pose_from_vec(final_vec[5:])
        return {cam_a: (rot_a, trans_a), cam_b: (rot_b, trans_b)}

    def _fuse_step_profiles(points_by_camera: dict[str, np.ndarray]) -> np.ndarray:
        if not bool(profile_fusion_cfg.get("enabled", True)):
            valid = [p for p in points_by_camera.values() if p.ndim == 2 and p.shape[0] > 0]
            return np.vstack(valid).astype(np.float64) if valid else np.empty((0, 3), dtype=np.float64)

        parts: list[np.ndarray] = []
        src: list[str] = []
        for cam_id, pts in points_by_camera.items():
            if pts.ndim == 2 and pts.shape[0] > 0:
                parts.append(pts.astype(np.float64))
                src.extend([cam_id] * pts.shape[0])
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        if len(parts) == 1:
            return parts[0]

        pts_all = np.vstack(parts).astype(np.float64)
        src_arr = np.asarray(src, dtype=object)

        center = pts_all.mean(axis=0)
        cov = np.cov((pts_all - center).T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        t_all = (pts_all - center) @ axis
        order = np.argsort(t_all)
        pts_all = pts_all[order]
        src_arr = src_arr[order]
        t_all = t_all[order]

        gap_mm = float(profile_fusion_cfg.get("gap_mm", 4.0))
        min_seg_points = int(profile_fusion_cfg.get("min_segment_points", 12))
        min_overlap_points = int(profile_fusion_cfg.get("min_overlap_points", 8))
        local_window_points = int(profile_fusion_cfg.get("local_window_points", 21))
        local_poly_degree = int(profile_fusion_cfg.get("local_poly_degree", 2))
        local_min_points = int(profile_fusion_cfg.get("local_min_points", 9))
        max_fuse_distance_mm = float(profile_fusion_cfg.get("max_fuse_distance_mm", 2.5))

        def _local_poly_smooth(curve_pts: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
            n = curve_pts.shape[0]
            if n < max(local_min_points, local_poly_degree + 2):
                return curve_pts
            win = max(local_poly_degree + 2, local_window_points)
            if win % 2 == 0:
                win += 1
            half = win // 2
            out = curve_pts.copy()
            for i in range(n):
                s0 = max(0, i - half)
                s1 = min(n, i + half + 1)
                # Expand near borders to keep enough support.
                if s1 - s0 < local_poly_degree + 2:
                    if s0 == 0:
                        s1 = min(n, s0 + local_poly_degree + 2)
                    else:
                        s0 = max(0, s1 - (local_poly_degree + 2))
                tt = t_vals[s0:s1]
                if tt.shape[0] < local_poly_degree + 2:
                    continue
                t0 = float(tt.mean())
                tc = tt - t0
                for ax in range(3):
                    yy = curve_pts[s0:s1, ax]
                    try:
                        coeff = np.polyfit(tc, yy, deg=local_poly_degree)
                        out[i, ax] = float(np.polyval(coeff, t_vals[i] - t0))
                    except Exception:
                        pass
            return out
        d = np.diff(t_all)
        split_idx = np.where(d > gap_mm)[0] + 1
        bounds = np.concatenate(([0], split_idx, [pts_all.shape[0]]))

        fused_segments: list[np.ndarray] = []
        for i in range(len(bounds) - 1):
            s0, s1 = int(bounds[i]), int(bounds[i + 1])
            seg_pts = pts_all[s0:s1]
            seg_src = src_arr[s0:s1]
            if seg_pts.shape[0] < min_seg_points:
                fused_segments.append(seg_pts)
                continue

            cams = np.unique(seg_src).tolist()
            if len(cams) < 2:
                fused_segments.append(seg_pts)
                continue
            cam_a, cam_b = cams[0], cams[1]
            a = seg_pts[seg_src == cam_a]
            b = seg_pts[seg_src == cam_b]
            if a.shape[0] < min_overlap_points or b.shape[0] < min_overlap_points:
                fused_segments.append(seg_pts)
                continue

            ta = (a - center) @ axis
            tb = (b - center) @ axis
            oa = np.argsort(ta)
            ob = np.argsort(tb)
            a, ta = a[oa], ta[oa]
            b, tb = b[ob], tb[ob]

            t0 = max(float(ta.min()), float(tb.min()))
            t1 = min(float(ta.max()), float(tb.max()))
            if t1 <= t0:
                fused_segments.append(seg_pts)
                continue

            n = max(a.shape[0], b.shape[0])
            t_grid = np.linspace(t0, t1, n, dtype=np.float64)
            ai = np.column_stack([np.interp(t_grid, ta, a[:, k]) for k in range(3)])
            bi = np.column_stack([np.interp(t_grid, tb, b[:, k]) for k in range(3)])
            # Conservative overlap fusion:
            # - If both cameras agree locally, average them.
            # - If they diverge, keep both points to avoid collapsing geometry.
            delta = np.linalg.norm(ai - bi, axis=1)
            agree = delta <= max_fuse_distance_mm
            fused_overlap_mid = 0.5 * (ai + bi)
            fused_overlap_mid = _local_poly_smooth(fused_overlap_mid, t_grid)

            overlap_parts: list[np.ndarray] = []
            if np.any(agree):
                overlap_parts.append(fused_overlap_mid[agree])
            if np.any(~agree):
                overlap_parts.append(ai[~agree])
                overlap_parts.append(bi[~agree])
            fused_overlap = (
                np.vstack(overlap_parts).astype(np.float64)
                if overlap_parts
                else np.empty((0, 3), dtype=np.float64)
            )

            # Keep non-overlapping tails to preserve mono-camera visibility.
            tail_a = a[(ta < t0) | (ta > t1)]
            tail_b = b[(tb < t0) | (tb > t1)]
            fused_seg = np.vstack([fused_overlap, tail_a, tail_b]).astype(np.float64)
            fused_segments.append(fused_seg)

        if not fused_segments:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([seg for seg in fused_segments if seg.shape[0] > 0]).astype(np.float64)

    # ------------------------------------------------------------------ #
    # Load calibration
    # ------------------------------------------------------------------ #
    configured_camera_ids = camera_ids(config)
    camera_models = {}
    try:
        for camera_id in configured_camera_ids:
            camera_models[camera_id] = load_camera_model(config, camera_id)
    except (CalibrationError, ValueError) as exc:
        _go_error(exc)
        raise

    # ------------------------------------------------------------------ #
    # SCANNING phase
    # ------------------------------------------------------------------ #
    try:
        sm.transition(ScannerState.SCANNING)
    except (ValueError, HardwareError) as exc:
        _go_error(exc)
        raise

    frames_by_camera: dict[str, list[np.ndarray]] = {}
    try:
        _run_pre_scan_exposure_calibration()

        def _capture_progress(step: int, total: int) -> None:
            _progress(step, total, f"Capturing step {step}/{total}")

        frames_by_camera = run_capture_sequence_multi(
            n_steps,
            config,
            progress_callback=_capture_progress,
        )
        for camera_id in frames_by_camera:
            latest = os.path.join("/tmp/scan_frames", f"latest_{camera_id}.jpg")
            _artifact(
                f"extract_{camera_id}",
                latest,
                f"Extraction {camera_id}",
                "image/jpeg",
            )
    except (HardwareError, Exception) as exc:
        _go_error(exc)
        raise

    # ------------------------------------------------------------------ #
    # PROCESSING phase
    # ------------------------------------------------------------------ #
    try:
        sm.transition(ScannerState.PROCESSING)
    except (ValueError, HardwareError) as exc:
        _go_error(exc)
        raise

    profiles: list[np.ndarray] = []
    profiles_by_camera: dict[str, list[np.ndarray]] = {}
    angle_step_rad = 2.0 * math.pi / n_steps
    extracted_by_camera: dict[str, list[dict]] = {}

    try:
        total_processing = max(1, sum(len(frames) for frames in frames_by_camera.values()))
        processed = 0
        for camera_id, frames in frames_by_camera.items():
            if camera_id not in camera_models:
                logger.warning("No calibration model loaded for camera %s", camera_id)
                continue
            cam_cfg = next(
                (
                    item
                    for item in config.get("cameras", [])
                    if str(item.get("id")) == str(camera_id)
                ),
                {},
            )
            threshold = int(cam_cfg.get("laser_threshold", default_threshold))
            mask_rects = cam_cfg.get("laser_mask", []) or []
            sampling = _sampling_config(cam_cfg)
            _camera_matrix, _dist_coeffs, _laser_plane, _cam_rot, _cam_trans = camera_models[camera_id]
            for idx, frame in enumerate(frames):
                angle_rad = idx * angle_step_rad
                line_px = extract_laser_line(
                    frame,
                    threshold=threshold,
                    min_pixels=min_pixels,
                    subpixel=subpixel,
                    mode=extraction_mode,
                    camera_id=camera_id,
                    mask_rects=mask_rects,
                    **sampling,
                )
                extracted_by_camera.setdefault(camera_id, []).append(
                    {"angle_rad": float(angle_rad), "line_pixels": line_px}
                )
                processed += 1
                _progress(
                    processed,
                    total_processing,
                    f"Extracting {camera_id} frame {idx + 1}/{len(frames)}",
                )

        tuned_poses = _run_auto_cheat_extrinsics(extracted_by_camera)
        _progress(processed, total_processing, "Fitting shared camera frame")
        for camera_id, (rot, trans) in tuned_poses.items():
            cm, dc, lp, _old_rot, _old_trans = camera_models[camera_id]
            camera_models[camera_id] = (cm, dc, lp, rot, trans)

        processed = 0
        triangulated_by_camera: dict[str, list[np.ndarray]] = {}
        for camera_id, extracted_items in extracted_by_camera.items():
            if camera_id not in camera_models:
                continue
            camera_matrix, dist_coeffs, laser_plane, cam_rot, cam_trans = camera_models[camera_id]
            for idx, item in enumerate(extracted_items):
                line_px = item["line_pixels"]
                if line_px.shape[0] > 0:
                    pts_3d = triangulate(
                        line_px,
                        camera_matrix,
                        dist_coeffs,
                        laser_plane,
                        float(item["angle_rad"]),
                        axis_point=_axis_point,
                        camera_to_platform_rotation=cam_rot,
                        camera_to_platform_translation=cam_trans,
                    )
                    if pts_3d.shape[0] > 0:
                        triangulated_by_camera.setdefault(camera_id, []).append(pts_3d)
                        profiles_by_camera.setdefault(camera_id, []).append(pts_3d)
                    else:
                        triangulated_by_camera.setdefault(camera_id, []).append(
                            np.empty((0, 3), dtype=np.float64)
                        )
                else:
                    triangulated_by_camera.setdefault(camera_id, []).append(
                        np.empty((0, 3), dtype=np.float64)
                    )
                processed += 1
                _progress(
                    processed,
                    total_processing,
                    f"Triangulating {camera_id} frame {idx + 1}/{len(extracted_items)}",
                )

        _progress(total_processing, total_processing, "Fusing half-turn duplicate profiles")
        profiles_by_camera = {}
        for camera_id, camera_profiles in triangulated_by_camera.items():
            camera_fused = fuse_half_turn_profiles(
                camera_profiles,
                n_steps=n_steps,
                enabled=bool(half_turn_fusion_cfg.get("enabled", True)),
                offset_tolerance_steps=int(half_turn_fusion_cfg.get("offset_tolerance_steps", 1)),
                max_pair_distance_mm=float(half_turn_fusion_cfg.get("max_pair_distance_mm", 6.0)),
                min_profile_points=int(half_turn_fusion_cfg.get("min_profile_points", 8)),
                distance_axes=str(half_turn_fusion_cfg.get("distance_axes", "xyz")),
            )
            profiles_by_camera[camera_id] = [
                p for p in camera_fused if p.ndim == 2 and p.shape[0] > 0
            ]

        max_steps = 0
        for arr in triangulated_by_camera.values():
            max_steps = max(max_steps, len(arr))
        _progress(total_processing, total_processing, "Applying profile regression")
        for step_idx in range(max_steps):
            step_map: dict[str, np.ndarray] = {}
            for cam_id, arr in triangulated_by_camera.items():
                if step_idx < len(arr):
                    step_map[cam_id] = arr[step_idx]
            fused = _fuse_step_profiles(step_map)
            if fused.shape[0] > 0:
                profiles.append(fused)
        profiles = fuse_half_turn_profiles(
            profiles,
            n_steps=n_steps,
            enabled=bool(half_turn_fusion_cfg.get("enabled", True)),
            offset_tolerance_steps=int(half_turn_fusion_cfg.get("offset_tolerance_steps", 1)),
            max_pair_distance_mm=float(half_turn_fusion_cfg.get("max_pair_distance_mm", 6.0)),
            min_profile_points=int(half_turn_fusion_cfg.get("min_profile_points", 8)),
            distance_axes=str(half_turn_fusion_cfg.get("distance_axes", "xyz")),
        )

        _progress(total_processing, total_processing, "Fusing per-step profiles")
        camera_clouds: dict[str, np.ndarray] = {}
        for camera_id in frames_by_camera:
            camera_profiles = profiles_by_camera.get(camera_id, [])
            if camera_profiles:
                _progress(total_processing, total_processing, f"Merging cloud {camera_id}")
                camera_cloud = merge_profiles(camera_profiles)
                if camera_cloud.shape[0] >= 20:
                    _progress(total_processing, total_processing, f"Filtering outliers {camera_id}")
                    camera_cloud = filter_outliers(
                        camera_cloud,
                        nb_neighbors=nb_neighbors,
                        std_ratio=std_ratio,
                    )
                camera_clouds[camera_id] = camera_cloud
                camera_cloud_path = os.path.join(output_dir, f"scan_{timestamp}_cloud_{camera_id}.ply")
                export_point_cloud_ply(camera_cloud, camera_cloud_path)
                _artifact(
                    f"cloud_{camera_id}",
                    camera_cloud_path,
                    f"Nuage {camera_id}",
                    "model/ply",
                    points=camera_cloud.shape[0],
                )

        _progress(total_processing, total_processing, "Merging combined cloud")
        cloud = merge_profiles(profiles)

        if cloud.shape[0] >= 20:
            _progress(total_processing, total_processing, "Filtering combined outliers")
            cloud = filter_outliers(cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        else:
            logger.warning("Too few points (%d) for outlier filtering", cloud.shape[0])

        mesh_clip_plane: np.ndarray | None = None
        if bool(top_plane_clip_cfg.get("enabled", False)):
            reference_camera = str(top_plane_clip_cfg.get("reference_camera", "right"))
            reference_cloud = camera_clouds.get(reference_camera)
            if reference_cloud is None:
                logger.warning("top_plane_clip: reference camera %s cloud missing", reference_camera)
            else:
                _progress(total_processing, total_processing, "Detecting top clipping plane")
                mesh_clip_plane = detect_top_plane(
                    reference_cloud,
                    enabled=True,
                    top_quantile=float(top_plane_clip_cfg.get("top_quantile", 0.90)),
                    bin_height_mm=float(top_plane_clip_cfg.get("bin_height_mm", 1.0)),
                    min_xz_extent_mm=float(top_plane_clip_cfg.get("min_xz_extent_mm", 20.0)),
                    min_density_ratio=float(top_plane_clip_cfg.get("min_density_ratio", 0.35)),
                    max_plane_thickness_mm=float(
                        top_plane_clip_cfg.get("max_plane_thickness_mm", 2.0)
                    ),
                    min_plane_points=int(top_plane_clip_cfg.get("min_plane_points", 80)),
                )
                if mesh_clip_plane is not None:
                    logger.info(
                        "top_plane_clip: applied plane=[%.4f, %.4f, %.4f, %.3f]",
                        mesh_clip_plane[0],
                        mesh_clip_plane[1],
                        mesh_clip_plane[2],
                        mesh_clip_plane[3],
                    )

        if bool(flat_caps_cfg.get("enabled", False)):
            _progress(total_processing, total_processing, "Adding flat caps")
            cloud = add_flat_caps_aligned(
                cloud,
                enabled=True,
                axis_mode=str(flat_caps_cfg.get("axis_mode", "pca")),
                axis_index=int(flat_caps_cfg.get("axis_index", 2)),
                grid_mm=float(flat_caps_cfg.get("grid_mm", 0.8)),
                top_quantile=float(flat_caps_cfg.get("top_quantile", 0.99)),
                bottom_quantile=float(flat_caps_cfg.get("bottom_quantile", 0.01)),
                border_pad_mm=float(flat_caps_cfg.get("border_pad_mm", 1.0)),
            )

        logger.info("Processing complete: %d 3D points", cloud.shape[0])

    except Exception as exc:
        _go_error(exc)
        raise RuntimeError(f"Processing phase failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # EXPORTING phase
    # ------------------------------------------------------------------ #
    try:
        sm.transition(ScannerState.EXPORTING)
    except (ValueError, HardwareError) as exc:
        _go_error(exc)
        raise

    try:
        _progress(n_steps, n_steps, "Exporting combined cloud")
        export_point_cloud_ply(cloud, cloud_path)
        _artifact(
            "cloud_combined",
            cloud_path,
            "Nuage combine",
            "model/ply",
            points=cloud.shape[0],
        )
        logger.info("Raw point cloud exported to %s", cloud_path)

        _progress(n_steps, n_steps, "Building mesh (Poisson)")
        mesh_poisson_cfg = dict(poisson_cfg)
        if "mesh_clip_plane" not in mesh_poisson_cfg:
            try:
                if "mesh_clip_plane" in locals() and mesh_clip_plane is not None:
                    mesh_poisson_cfg["mesh_clip_plane"] = mesh_clip_plane.tolist()
                    mesh_poisson_cfg.setdefault(
                        "mesh_clip_margin_mm",
                        float(top_plane_clip_cfg.get("clip_margin_mm", 1.0)),
                    )
                    mesh_poisson_cfg.setdefault(
                        "mesh_clip_cap",
                        bool(top_plane_clip_cfg.get("mesh_clip_cap", True)),
                    )
            except NameError:
                pass

        if fmt == "obj":
            export_obj(cloud, output_path, poisson=mesh_poisson_cfg)
        else:
            export_stl(cloud, output_path, poisson=mesh_poisson_cfg)

        _artifact("mesh", output_path, "STL final" if fmt == "stl" else "OBJ final", f"model/{fmt}")

        logger.info("Scan exported to %s", output_path)

    except Exception as exc:
        _go_error(exc)
        raise RuntimeError(f"Export failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # COMPLETE
    # ------------------------------------------------------------------ #
    try:
        sm.transition(ScannerState.COMPLETE)
    except (ValueError, HardwareError) as exc:
        logger.warning("Could not set COMPLETE state: %s", exc)

    _progress(n_steps, n_steps, "Scan complete!")
    return os.path.abspath(output_path)
