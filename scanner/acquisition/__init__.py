"""scanner.acquisition — Image capture pipeline.

Exposes run_capture_sequence which drives the motor, laser and camera
to acquire one frame per step over a full 360° rotation.
"""

import logging
import math
import os
from typing import Callable, Optional

import numpy as np

from scanner.hardware import (
    HardwareError,
    camera_capture,
    laser_set,
    motor_step,
)

logger = logging.getLogger(__name__)

_FRAME_DIR = "/tmp/scan_frames"


def _save_frame(frame: np.ndarray, step_idx: int, config: dict) -> None:
    """Save *frame* to disk with laser-line overlay."""
    try:
        import cv2
        from scanner.calibration import load_background_filter
        from scanner.processing import (
            crop_laser_line,
            extract_laser_line,
            fill_occluded_laser_gaps,
        )

        os.makedirs(_FRAME_DIR, exist_ok=True)

        proc_cfg = config.get("processing", {})
        threshold = int(proc_cfg.get("laser_threshold", 180))
        min_px = int(proc_cfg.get("min_line_pixels", 10))
        subpixel = bool(proc_cfg.get("subpixel", True))
        extraction_mode = str(proc_cfg.get("extraction_mode", "component_axis"))
        occlusion_interpolation = bool(proc_cfg.get("occlusion_interpolation", True))
        occlusion_max_gap_rows_raw = int(proc_cfg.get("occlusion_max_gap_rows", 0))
        occlusion_max_gap_rows = (
            occlusion_max_gap_rows_raw if occlusion_max_gap_rows_raw > 0 else None
        )
        background_filter = load_background_filter()
        crop_left_of_col = (
            float(background_filter["crop_left_of_col"])
            if background_filter.get("enabled") and background_filter.get("crop_left_of_col") is not None
            else None
        )

        overlay = frame.copy()
        try:
            line = extract_laser_line(
                frame,
                threshold=threshold,
                min_pixels=min_px,
                subpixel=subpixel,
                mode=extraction_mode,
            )
            line = crop_laser_line(line, crop_left_of_col=crop_left_of_col, min_points=min_px)
            if occlusion_interpolation and line.shape[0] > 0:
                line = fill_occluded_laser_gaps(
                    line,
                    image_height=frame.shape[0],
                    max_gap_rows=occlusion_max_gap_rows,
                    min_points=min_px,
                )
            for i in range(line.shape[0]):
                col, row = int(round(line[i, 0])), int(round(line[i, 1]))
                cv2.circle(overlay, (col, row), 1, (0, 0, 255), -1)
        except Exception:
            pass  # Save frame without overlay if extraction fails

        path = os.path.join(_FRAME_DIR, f"frame_{step_idx:03d}.jpg")
        cv2.imwrite(path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # Also write a "latest" symlink-equivalent (overwrite)
        latest = os.path.join(_FRAME_DIR, "latest.jpg")
        cv2.imwrite(latest, overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
        n_steps: Total number of rotational steps (e.g. 200 for 360°).
        config: Full settings dict loaded from settings.yaml.
        progress_callback: Optional callable(current_step, total_steps)
            called after each captured frame.
        save_frames: If True, save each frame as JPEG to /tmp/scan_frames/.

    Returns:
        List of *n_steps* BGR images (numpy arrays, shape H×W×3, dtype uint8).

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
            # 1. Advance by steps_per_photo motor steps
            motor_step(steps_per_photo, direction)

            # 2. Laser on
            laser_set(True)

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
