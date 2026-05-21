"""USB camera driver using OpenCV VideoCapture."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class USBCamera:
    """OpenCV-backed USB camera.

    Args:
        config: Camera configuration dict. Recognised keys include
            ``device_index``, ``resolution``, ``exposure_us`` and ``gain``.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        try:
            import cv2  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("opencv-python not available for USB camera") from exc

        self._cv2 = cv2
        self._device_index = int(config.get("device_index", config.get("index", 0)))
        res = config.get("resolution", [640, 480])
        self._width = int(res[0])
        self._height = int(res[1])
        self._exposure_us = int(config.get("exposure_us", 1000))
        self._gain = float(config.get("gain", 1.0))

        backend = config.get("backend")
        if backend:
            backend_value = getattr(cv2, str(backend), None)
            self._cap = cv2.VideoCapture(self._device_index, backend_value) if backend_value else cv2.VideoCapture(self._device_index)
        else:
            self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            raise HardwareError(f"USB camera {self._device_index} could not be opened")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self.set_exposure(self._exposure_us, self._gain)

        logger.info(
            "USBCamera initialised (index=%d, %dx%d)",
            self._device_index,
            self._width,
            self._height,
        )

    def capture(self) -> np.ndarray:
        """Capture one BGR frame."""
        from scanner.hardware import HardwareError

        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise HardwareError(f"USB camera {self._device_index} capture failed")
        return frame

    def set_exposure(self, exposure_us: int, gain: Optional[float] = None) -> None:
        """Best-effort manual exposure update.

        Many recovered USB cameras expose non-standard controls. Failures are
        logged but do not abort initialisation because capture may still work.
        """
        self._exposure_us = int(exposure_us)
        if gain is not None:
            self._gain = float(gain)
        try:
            self._cap.set(self._cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # OpenCV backends commonly use seconds or log2 units. Store the
            # requested value anyway and let hardware validation decide.
            self._cap.set(self._cv2.CAP_PROP_EXPOSURE, float(self._exposure_us) / 1_000_000.0)
            self._cap.set(self._cv2.CAP_PROP_GAIN, self._gain)
        except Exception as exc:
            logger.warning("USB camera exposure update failed: %s", exc)

    def close(self) -> None:
        """Release the camera device."""
        try:
            self._cap.release()
        except Exception as exc:
            logger.warning("Error closing USB camera: %s", exc)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
