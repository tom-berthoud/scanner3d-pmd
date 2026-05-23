"""USB camera driver using OpenCV VideoCapture."""

import logging
import shutil
import subprocess
import time
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
        self._device_path = config.get("device_path")
        res = config.get("resolution", [640, 480])
        self._width = int(res[0])
        self._height = int(res[1])
        self._exposure_us = int(config.get("exposure_us", 1000))
        self._gain = float(config.get("gain", 1.0))
        self._pixel_format = config.get("pixel_format")
        self._focus_absolute = config.get("focus_absolute")
        self._flush_frames = int(config.get("flush_frames", 3))
        self._v4l2_exposure_scale_us = int(config.get("v4l2_exposure_scale_us", 100))
        self._capture_device = str(self._device_path) if self._device_path else self._device_index
        self._v4l2_ctrl_names: set[str] = set()

        self._backend = config.get("backend")
        self._open_capture()

        logger.info(
            "USBCamera initialised (device=%s, %dx%d)",
            self._capture_device,
            self._width,
            self._height,
        )

    def _open_capture(self) -> None:
        """Open VideoCapture and apply requested USB camera properties."""
        from scanner.hardware import HardwareError

        cv2 = self._cv2
        self._set_v4l2_format()
        if self._backend:
            backend_value = getattr(cv2, str(self._backend), None)
            self._cap = (
                cv2.VideoCapture(self._capture_device, backend_value)
                if backend_value
                else cv2.VideoCapture(self._capture_device)
            )
        else:
            self._cap = cv2.VideoCapture(self._capture_device)

        if not self._cap.isOpened():
            raise HardwareError(f"USB camera {self._capture_device} could not be opened")

        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if self._pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*str(self._pixel_format)[:4])
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._v4l2_ctrl_names = self._list_v4l2_control_names()
        self.set_exposure(self._exposure_us, self._gain)
        if self._focus_absolute not in (None, ""):
            self._apply_v4l2_controls({"focus_absolute": int(float(self._focus_absolute))})
        self._flush_buffer()

    def _reopen_capture(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        time.sleep(0.05)
        self._open_capture()

    def _flush_buffer(self) -> None:
        for _ in range(max(0, self._flush_frames)):
            try:
                self._cap.grab()
            except Exception:
                break

    def _apply_v4l2_controls(self, controls: dict[str, int]) -> dict[str, str]:
        """Best-effort apply UVC/V4L2 controls with v4l2-ctl."""
        if not self._device_path or not shutil.which("v4l2-ctl"):
            return {}
        applied = {}
        for key, value in controls.items():
            cmd = ["v4l2-ctl", "-d", str(self._device_path), "-c", f"{key}={value}"]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2.0)
                applied[key] = str(value)
            except Exception as exc:
                logger.debug("v4l2 control %s=%s failed: %s", key, value, exc)
        return applied

    def _set_v4l2_format(self) -> None:
        """Force the requested V4L2 stream mode before OpenCV opens the device."""
        if not self._device_path or not shutil.which("v4l2-ctl"):
            return
        fmt = f"width={self._width},height={self._height}"
        if self._pixel_format:
            fmt = f"{fmt},pixelformat={str(self._pixel_format)[:4]}"
        cmd = ["v4l2-ctl", "-d", str(self._device_path), f"--set-fmt-video={fmt}"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2.0)
        except Exception as exc:
            logger.debug("v4l2 format request failed (%s): %s", fmt, exc)

    def _list_v4l2_control_names(self) -> set[str]:
        if not self._device_path or not shutil.which("v4l2-ctl"):
            return set()
        try:
            proc = subprocess.run(
                ["v4l2-ctl", "-d", str(self._device_path), "--list-ctrls-menus"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except Exception:
            return set()
        names = set()
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if not stripped or " " not in stripped:
                continue
            names.add(stripped.split()[0])
        return names

    def _control_exists(self, name: str) -> bool:
        return not self._v4l2_ctrl_names or name in self._v4l2_ctrl_names

    def _get_v4l2_controls(self) -> dict[str, str]:
        if not self._device_path or not shutil.which("v4l2-ctl"):
            return {}
        result = {}
        for key in (
            "auto_exposure",
            "exposure_auto",
            "exposure_auto_priority",
            "exposure_absolute",
            "exposure_time_absolute",
            "gain",
            "white_balance_automatic",
            "white_balance_temperature_auto",
            "focus_auto",
            "focus_absolute",
        ):
            cmd = ["v4l2-ctl", "-d", str(self._device_path), "--get-ctrl", key]
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2.0)
                result[key] = proc.stdout.strip()
            except Exception:
                continue
        return result

    def _get_v4l2_format(self) -> str | None:
        if not self._device_path or not shutil.which("v4l2-ctl"):
            return None
        try:
            proc = subprocess.run(
                ["v4l2-ctl", "-d", str(self._device_path), "--get-fmt-video"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            return proc.stdout.strip()
        except Exception:
            return None

    def capture(self) -> np.ndarray:
        """Capture one BGR frame."""
        from scanner.hardware import HardwareError

        self._flush_buffer()
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
        exposure_abs = max(1, int(round(self._exposure_us / max(1, self._v4l2_exposure_scale_us))))
        v4l2_controls: dict[str, int] = {}
        if self._control_exists("auto_exposure"):
            v4l2_controls["auto_exposure"] = 1
        if self._control_exists("exposure_auto"):
            v4l2_controls["exposure_auto"] = 1
        if self._control_exists("exposure_auto_priority"):
            v4l2_controls["exposure_auto_priority"] = 0
        if self._control_exists("white_balance_automatic"):
            v4l2_controls["white_balance_automatic"] = 0
        if self._control_exists("white_balance_temperature_auto"):
            v4l2_controls["white_balance_temperature_auto"] = 0
        if self._control_exists("exposure_absolute"):
            v4l2_controls["exposure_absolute"] = exposure_abs
        if self._control_exists("exposure_time_absolute"):
            v4l2_controls["exposure_time_absolute"] = exposure_abs
        if self._control_exists("gain"):
            v4l2_controls["gain"] = max(0, int(round(self._gain)))
        self._apply_v4l2_controls(v4l2_controls)

    def set_controls(self, controls: dict) -> dict:
        """Apply runtime USB camera controls and return driver-reported info."""
        reopen = False
        if controls.get("pixel_format"):
            self._pixel_format = str(controls["pixel_format"])[:4]
            reopen = True
        if controls.get("width") is not None and controls.get("height") is not None:
            new_width = int(controls["width"])
            new_height = int(controls["height"])
            reopen = reopen or new_width != self._width or new_height != self._height
            self._width = new_width
            self._height = new_height
        if controls.get("focus_absolute") not in (None, ""):
            self._focus_absolute = int(float(controls["focus_absolute"]))
            self._apply_v4l2_controls({"focus_auto": 0, "focus_absolute": self._focus_absolute})
        elif controls.get("lens_position") not in (None, ""):
            self._focus_absolute = int(float(controls["lens_position"]))
            self._apply_v4l2_controls({"focus_auto": 0, "focus_absolute": self._focus_absolute})
        if reopen:
            self._reopen_capture()
        elif self._pixel_format:
            fourcc = self._cv2.VideoWriter_fourcc(*self._pixel_format)
            self._cap.set(self._cv2.CAP_PROP_FOURCC, fourcc)
            self._cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        exposure = controls.get("exposure_us", self._exposure_us)
        gain = controls.get("gain", self._gain)
        self.set_exposure(int(exposure), float(gain))
        self._flush_buffer()
        return self.get_info()

    def get_info(self) -> dict:
        """Return requested and OpenCV-reported USB camera settings."""
        fourcc_value = int(self._cap.get(self._cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()
        return {
            "driver": "USBCamera",
            "requested": {
                "width": self._width,
                "height": self._height,
                "exposure_us": self._exposure_us,
                "gain": self._gain,
                "pixel_format": self._pixel_format,
                "focus_absolute": self._focus_absolute,
                "flush_frames": self._flush_frames,
                "device_index": self._device_index,
                "device_path": self._device_path,
            },
            "actual": {
                "width": int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT)),
                "exposure": self._cap.get(self._cv2.CAP_PROP_EXPOSURE),
                "gain": self._cap.get(self._cv2.CAP_PROP_GAIN),
                "fourcc": fourcc,
                "v4l2_format": self._get_v4l2_format(),
                "v4l2_controls": self._get_v4l2_controls(),
            },
        }

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
