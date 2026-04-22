"""scanner.hardware.camera — Pi Camera Module 3 interface.

Wraps the picamera2 library to provide a simple capture() method.
Autofocus is locked to manual mode on initialisation so that the camera
matrix calibration remains valid for the entire scan session.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PiCamera:
    """Pi Camera Module 3 driver using picamera2.

    Focus is locked in manual mode on init to prevent autofocus from
    invalidating the calibration (see agents.md section 2).

    Args:
        config: Camera configuration dict with keys:
            - resolution: [width, height], default [1920, 1080]
            - exposure_us: shutter time in microseconds, default 5000
            - gain: analogue gain, default 1.0
            - awb_mode: 'off' or 'auto', default 'off'
            - awb_gains: [red_gain, blue_gain], default [1.5, 1.2]

    Raises:
        HardwareError: if picamera2 is unavailable or the camera fails to open.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        try:
            from picamera2 import Picamera2  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("picamera2 not available — install it on the Pi") from exc

        res = config.get("resolution", [1920, 1080])
        self._width: int = int(res[0])
        self._height: int = int(res[1])
        self._exposure_us: int = int(config.get("exposure_us", 5000))
        self._gain: float = float(config.get("gain", 1.0))
        self._awb_mode: str = str(config.get("awb_mode", "off"))
        awb_gains = config.get("awb_gains", [1.5, 1.2])
        self._awb_gains: tuple[float, float] = (float(awb_gains[0]), float(awb_gains[1]))

        try:
            self._cam = Picamera2()
            capture_cfg = self._cam.create_still_configuration(
                main={"size": (self._width, self._height), "format": "BGR888"},
            )
            self._cam.configure(capture_cfg)
            self._cam.start()

            # Lock focus in manual mode — non-negotiable (agents.md §2)
            # AfMode 0 = manual; LensPosition 4.0 ≈ 25 cm focus distance
            self._cam.set_controls(
                {
                    "AfMode": 0,
                    "LensPosition": 2.53,
                    "ExposureTime": self._exposure_us,
                    "AnalogueGain": self._gain,
                }
            )
            if self._awb_mode == "off":
                self._cam.set_controls(
                    {
                        "AwbEnable": False,
                        "ColourGains": self._awb_gains,
                    }
                )
            else:
                self._cam.set_controls({"AwbEnable": True})

            logger.info(
                "PiCamera initialised (%dx%d, exposure=%d µs, gain=%.2f, awb=%s)",
                self._width,
                self._height,
                self._exposure_us,
                self._gain,
                self._awb_mode,
            )
        except Exception as exc:
            raise HardwareError(f"Camera init failed: {exc}") from exc

    def capture(self) -> np.ndarray:
        """Capture a single frame.

        Returns:
            BGR image as numpy array of shape (H, W, 3), dtype uint8.

        Raises:
            HardwareError: if capture fails.
        """
        from scanner.hardware import HardwareError

        try:
            frame = self._cam.capture_array("main")
            logger.debug("PiCamera.capture() → %s", frame.shape)
            return frame
        except Exception as exc:
            raise HardwareError(f"Camera capture failed: {exc}") from exc

    def close(self) -> None:
        """Release the camera resource."""
        try:
            self._cam.stop()
            self._cam.close()
            logger.info("PiCamera closed")
        except Exception as exc:
            logger.warning("Error closing camera: %s", exc)

    def __del__(self) -> None:
        """Attempt to release the camera on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
