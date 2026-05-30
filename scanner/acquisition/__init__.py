"""scanner.acquisition â€” Image capture pipeline.

Exposes run_capture_sequence which drives the motor, laser and camera
to acquire one frame per step over a full 360Â° rotation.
"""

import logging
import os
from typing import Callable, Optional

import numpy as np

from scanner.hardware import (
    HardwareError,
    camera_capture,
    camera_capture_all,
    check_door_interlock,
    laser_set,
    motor_step,
)

logger = logging.getLogger(__name__)

_FRAME_DIR = "/tmp/scan_frames"


def _sampling_config(cam_cfg: dict | None = None) -> dict[str, int]:
    sampling = (cam_cfg or {}).get("laser_sampling", {}) or {}
    return {
        "x_stride": max(1, int(sampling.get("x_stride", 1))),
        "y_stride": max(1, int(sampling.get("y_stride", 1))),
        "x_offset": max(0, int(sampling.get("x_offset", 0))),
        "y_offset": max(0, int(sampling.get("y_offset", 0))),
    }


def _camera_processing_config(
    config: dict,
    camera_id: str | None = None,
) -> tuple[int, list, dict[str, int]]:
    proc_cfg = config.get("processing", {})
    threshold = int(proc_cfg.get("laser_threshold", 180))
    mask = []
    sampling = _sampling_config()
    if camera_id is not None:
        try:
            from scanner.calibration import camera_config_by_id

            cam_cfg = camera_config_by_id(config, str(camera_id))
            threshold = int(cam_cfg.get("laser_threshold", threshold))
            mask = cam_cfg.get("laser_mask", []) or []
            sampling = _sampling_config(cam_cfg)
        except Exception:
            pass
    return threshold, mask, sampling


def _save_frame(
    frame: np.ndarray,
    step_idx: int,
    config: dict,
    camera_id: str | None = None,
) -> None:
    """Save *frame* to disk with laser-line overlay."""
    try:
        import cv2
        from scanner.processing import extract_laser_line

        os.makedirs(_FRAME_DIR, exist_ok=True)

        proc_cfg = config.get("processing", {})
        threshold, mask_rects, sampling = _camera_processing_config(config, camera_id)
        min_px = int(proc_cfg.get("min_line_pixels", 10))
        subpixel = bool(proc_cfg.get("subpixel", True))
        extraction_mode = str(proc_cfg.get("extraction_mode", "row_mean"))

        overlay = frame.copy()
        try:
            line = extract_laser_line(
                frame,
                threshold=threshold,
                min_pixels=min_px,
                subpixel=subpixel,
                mode=extraction_mode,
                camera_id=camera_id,
                mask_rects=mask_rects,
                **sampling,
            )
            for i in range(line.shape[0]):
                col, row = int(round(line[i, 0])), int(round(line[i, 1]))
                cv2.circle(overlay, (col, row), 1, (0, 0, 255), -1)
        except Exception:
            pass  # Save frame without overlay if extraction fails

        suffix = "" if camera_id is None else f"_{camera_id}"
        path = os.path.join(_FRAME_DIR, f"frame_{step_idx:03d}{suffix}.jpg")
        cv2.imwrite(path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # Also write a "latest" symlink-equivalent (overwrite)
        latest_name = "latest.jpg" if camera_id is None else f"latest_{camera_id}.jpg"
        latest = os.path.join(_FRAME_DIR, latest_name)
        cv2.imwrite(latest, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if camera_id is not None:
            cv2.imwrite(os.path.join(_FRAME_DIR, "latest.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception as exc:
        logger.debug("Could not save frame %d: %s", step_idx, exc)


def run_capture_sequence(
    n_steps: int,
    config: dict,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_frames: bool = True,
) -> list[np.ndarray]:
    """Drive the scanner through *n_steps* and collect one image per step.

    For each step the sequence is:
        1. Advance motor by one step
        2. Turn laser ON
        3. Capture image
        4. Turn laser OFF

    The laser is always left OFF at the end of the sequence, even on error.

    Args:
        n_steps: Total number of rotational steps (e.g. 200 for 360Â°).
        config: Full settings dict loaded from settings.yaml.
        progress_callback: Optional callable(current_step, total_steps)
            called after each captured frame.
        save_frames: If True, save each frame as JPEG to /tmp/scan_frames/.

    Returns:
        List of *n_steps* BGR images (numpy arrays, shape HÃ—WÃ—3, dtype uint8).

    Raises:
        HardwareError: if any hardware operation fails.
    """
    direction: str = config.get("scan", {}).get("direction", "clockwise")
    frames: list[np.ndarray] = []

    motor_cfg = config.get("motor", {})
    total_motor_steps = int(motor_cfg.get("steps_per_rev", 200)) * int(
        motor_cfg.get("microstepping", 1)
    )
    steps_per_photo = max(1, total_motor_steps // n_steps)

    logger.info(
        "Starting capture sequence: %d photos, %d motor steps/photo "
        "(%d total steps/rev), direction=%s",
        n_steps,
        steps_per_photo,
        total_motor_steps,
        direction,
    )

    # Clear previous frames
    if save_frames:
        try:
            import shutil
            if os.path.exists(_FRAME_DIR):
                shutil.rmtree(_FRAME_DIR)
        except Exception:
            pass

    try:
        for step_idx in range(n_steps):
            # 0. Safety door interlock â€” abort before energising the laser
            check_door_interlock()

            # 1. Advance by steps_per_photo motor steps
            motor_step(steps_per_photo, direction)

            # 2. Laser on
            laser_set(True)
            check_door_interlock()

            # 3. Capture
            frame = camera_capture()
            frames.append(frame)

            # 4. Laser off immediately after capture
            laser_set(False)

            if save_frames:
                _save_frame(frame, step_idx, config)

            if progress_callback is not None:
                progress_callback(step_idx + 1, n_steps)

            logger.debug("Step %d/%d captured (frame shape=%s)", step_idx + 1, n_steps, frame.shape)

    except HardwareError:
        # Ensure laser is off on any hardware error
        try:
            laser_set(False)
        except HardwareError as safety_err:
            logger.error("Could not turn off laser during error recovery: %s", safety_err)
        raise
    except Exception as exc:
        try:
            laser_set(False)
        except HardwareError as safety_err:
            logger.error("Could not turn off laser during error recovery: %s", safety_err)
        raise HardwareError(f"Capture sequence failed at step {len(frames)}: {exc}") from exc

    logger.info(
        "Capture sequence complete: %d frames collected", len(frames)
    )
    return frames


def run_capture_sequence_multi(
    n_steps: int,
    config: dict,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_frames: bool = True,
) -> dict[str, list[np.ndarray]]:
    """Drive the scanner and collect one image per camera at each step.

    The motor is moved once per step. The laser is then enabled, every camera
    is captured sequentially, and the laser is disabled immediately after the
    last capture.
    """
    direction: str = config.get("scan", {}).get("direction", "clockwise")
    frames_by_camera: dict[str, list[np.ndarray]] = {}

    motor_cfg = config.get("motor", {})
    total_motor_steps = int(motor_cfg.get("steps_per_rev", 200)) * int(
        motor_cfg.get("microstepping", 1)
    )
    steps_per_photo = max(1, total_motor_steps // n_steps)

    if save_frames:
        try:
            import shutil

            if os.path.exists(_FRAME_DIR):
                shutil.rmtree(_FRAME_DIR)
        except Exception:
            pass

    logger.info(
        "Starting multi-camera capture sequence: %d photos, %d motor steps/photo",
        n_steps,
        steps_per_photo,
    )

    try:
        for step_idx in range(n_steps):
            # Safety door interlock â€” abort before energising the laser
            check_door_interlock()
            motor_step(steps_per_photo, direction)
            laser_set(True)
            check_door_interlock()
            step_frames = camera_capture_all()
            laser_set(False)

            for camera_id, frame in step_frames.items():
                frames_by_camera.setdefault(camera_id, []).append(frame)
                if save_frames:
                    _save_frame(frame, step_idx, config, camera_id=camera_id)

            if progress_callback is not None:
                progress_callback(step_idx + 1, n_steps)

    except HardwareError:
        try:
            laser_set(False)
        except HardwareError as safety_err:
            logger.error("Could not turn off laser during error recovery: %s", safety_err)
        raise
    except Exception as exc:
        try:
            laser_set(False)
        except HardwareError as safety_err:
            logger.error("Could not turn off laser during error recovery: %s", safety_err)
        raise HardwareError(f"Multi-camera capture failed at step {step_idx}: {exc}") from exc

    logger.info(
        "Multi-camera capture complete: %s",
        {camera_id: len(frames) for camera_id, frames in frames_by_camera.items()},
    )
    return frames_by_camera


