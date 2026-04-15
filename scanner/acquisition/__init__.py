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
        from scanner.processing import extract_laser_line

        os.makedirs(_FRAME_DIR, exist_ok=True)

        proc_cfg = config.get("processing", {})
        threshold = int(proc_cfg.get("laser_threshold", 180))
        min_px = int(proc_cfg.get("min_line_pixels", 10))
        subpixel = bool(proc_cfg.get("subpixel", True))

        overlay = frame.copy()
        try:
            line = extract_laser_line(frame, threshold=threshold, min_pixels=min_px, subpixel=subpixel)
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

    logger.info("Starting capture sequence: %d steps, direction=%s", n_steps, direction)

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
            # 1. Advance one step
            motor_step(1, direction)

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
