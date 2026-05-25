"""scanner.orchestration.scan — Main scan loop.

Orchestrates the full acquisition → processing → reconstruction → export
pipeline, managing the state machine and LEDs throughout.
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
    from scanner.hardware import HardwareError, laser_set, led_set, led_blink
    from scanner.calibration import (
        CalibrationError,
        camera_ids,
        load_camera_model,
    )
    from scanner.acquisition import run_capture_sequence_multi
    from scanner.processing import extract_laser_line, triangulate
    from scanner.reconstruction import merge_profiles, filter_outliers
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

    proc_cfg = config.get("processing", {})
    default_threshold: int = int(proc_cfg.get("laser_threshold", 180))
    min_pixels: int = int(proc_cfg.get("min_line_pixels", 10))
    subpixel: bool = bool(proc_cfg.get("subpixel", True))
    extraction_mode: str = str(proc_cfg.get("extraction_mode", "row_mean"))

    recon_cfg = config.get("reconstruction", {})
    nb_neighbors: int = int(recon_cfg.get("outlier_nb_neighbors", 20))
    std_ratio: float = float(recon_cfg.get("outlier_std_ratio", 2.0))

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
            led_set("orange", False)
            led_set("red", True)
        except Exception as led_exc:
            logger.error("LED update during error: %s", led_exc)
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
        led_set("orange", True)
        led_set("red", False)
    except (ValueError, HardwareError) as exc:
        _go_error(exc)
        raise

    frames_by_camera: dict[str, list[np.ndarray]] = {}
    try:
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
        led_blink("orange", 4.0)  # fast blink during processing
    except (ValueError, HardwareError) as exc:
        _go_error(exc)
        raise

    profiles: list[np.ndarray] = []
    profiles_by_camera: dict[str, list[np.ndarray]] = {}
    angle_step_rad = 2.0 * math.pi / n_steps

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
            camera_matrix, dist_coeffs, laser_plane, cam_rot, cam_trans = camera_models[camera_id]
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
                if line_px.shape[0] > 0:
                    pts_3d = triangulate(
                        line_px,
                        camera_matrix,
                        dist_coeffs,
                        laser_plane,
                        angle_rad,
                        axis_point=_axis_point,
                        camera_to_platform_rotation=cam_rot,
                        camera_to_platform_translation=cam_trans,
                    )
                    profiles.append(pts_3d)
                    profiles_by_camera.setdefault(camera_id, []).append(pts_3d)
                processed += 1
                _progress(
                    processed,
                    total_processing,
                    f"Processing {camera_id} frame {idx + 1}/{len(frames)}",
                )

            camera_profiles = profiles_by_camera.get(camera_id, [])
            if camera_profiles:
                camera_cloud = merge_profiles(camera_profiles)
                if camera_cloud.shape[0] >= 20:
                    camera_cloud = filter_outliers(
                        camera_cloud,
                        nb_neighbors=nb_neighbors,
                        std_ratio=std_ratio,
                    )
                camera_cloud_path = os.path.join(output_dir, f"scan_{timestamp}_cloud_{camera_id}.ply")
                export_point_cloud_ply(camera_cloud, camera_cloud_path)
                _artifact(
                    f"cloud_{camera_id}",
                    camera_cloud_path,
                    f"Nuage {camera_id}",
                    "model/ply",
                    points=camera_cloud.shape[0],
                )

        cloud = merge_profiles(profiles)

        if cloud.shape[0] >= 20:
            cloud = filter_outliers(cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        else:
            logger.warning("Too few points (%d) for outlier filtering", cloud.shape[0])

        logger.info("Processing complete: %d 3D points", cloud.shape[0])

    except Exception as exc:
        _go_error(exc)
        raise RuntimeError(f"Processing phase failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # EXPORTING phase
    # ------------------------------------------------------------------ #
    try:
        sm.transition(ScannerState.EXPORTING)
        led_blink("orange", 0.5)  # slow blink during export
    except (ValueError, HardwareError) as exc:
        _go_error(exc)
        raise

    try:
        export_point_cloud_ply(cloud, cloud_path)
        _artifact(
            "cloud_combined",
            cloud_path,
            "Nuage combine",
            "model/ply",
            points=cloud.shape[0],
        )
        logger.info("Raw point cloud exported to %s", cloud_path)

        _progress(n_steps, n_steps, "Exporting mesh…")
        if fmt == "obj":
            export_obj(cloud, output_path, poisson=poisson_cfg)
        else:
            export_stl(cloud, output_path, poisson=poisson_cfg)

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
        led_set("orange", False)
        led_set("red", False)
    except (ValueError, HardwareError) as exc:
        logger.warning("Could not set COMPLETE state: %s", exc)

    _progress(n_steps, n_steps, "Scan complete!")
    return os.path.abspath(output_path)
