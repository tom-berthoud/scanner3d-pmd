"""scanner.hardware - Hardware abstraction layer.

The public API remains compatible with the original single-camera code while
also exposing camera-id aware helpers for the two-camera scanner layout.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class HardwareError(Exception):
    """Raised when a hardware operation fails or the hardware is unavailable."""


_ON_PI: bool = False
try:
    import gpiozero  # noqa: F401

    _ON_PI = True
    logger.info("gpiozero detected - using real hardware drivers")
except ImportError:
    logger.info("gpiozero not found - using mock hardware drivers")


_camera_instance = None
_camera_instances: dict[str, object] = {}
_motor_instance = None
_laser_instance = None
_led_instance = None
_display_instance = None


def init_hardware(config: dict) -> None:
    """Initialise all hardware singletons from *config*."""
    global _camera_instance, _camera_instances, _motor_instance, _laser_instance
    global _led_instance, _display_instance

    from scanner.calibration import camera_configs

    logger.info("Initialising hardware (on_pi=%s)", _ON_PI)
    iface_cfg = config.get("interface", {})
    display_type = str(iface_cfg.get("display_type", "oled")).lower()
    display_enabled = display_type not in ("none", "off", "disabled")

    try:
        if _ON_PI:
            from scanner.hardware.camera import PiCamera
            from scanner.hardware.display import Display
            from scanner.hardware.laser import Laser
            from scanner.hardware.led import LED
            from scanner.hardware.motor import StepperMotor
            from scanner.hardware.usb_camera import USBCamera

            _camera_instances = {}
            for cam_cfg in camera_configs(config):
                cam_id = str(cam_cfg.get("id", "main"))
                cam_type = str(cam_cfg.get("type", "pi")).lower()
                _camera_instances[cam_id] = (
                    USBCamera(cam_cfg) if cam_type == "usb" else PiCamera(cam_cfg)
                )
            _motor_instance = StepperMotor(config.get("motor", {}))
            _laser_instance = Laser(config.get("laser", {}))
            _led_instance = LED(iface_cfg)
            _display_instance = Display(iface_cfg) if display_enabled else None
        else:
            from scanner.hardware.mock import MockCamera, MockDisplay, MockLED, MockLaser, MockMotor

            _camera_instances = {
                str(cam_cfg.get("id", "main")): MockCamera(cam_cfg)
                for cam_cfg in camera_configs(config)
            }
            _motor_instance = MockMotor(config.get("motor", {}))
            _laser_instance = MockLaser(config.get("laser", {}))
            _led_instance = MockLED(iface_cfg)
            _display_instance = MockDisplay(iface_cfg) if display_enabled else None

        if not _camera_instances:
            raise HardwareError("No camera configured")
        _camera_instance = next(iter(_camera_instances.values()))
        if _display_instance is None:
            logger.info("Display disabled by config (interface.display_type=%s)", display_type)
    except HardwareError:
        raise
    except Exception as exc:
        raise HardwareError(f"Hardware initialisation failed: {exc}") from exc


def camera_capture(camera_id: str | None = None) -> np.ndarray:
    """Capture one frame from one camera.

    If *camera_id* is omitted, the first configured camera is used for legacy
    callers.
    """
    cam = _camera_instance if camera_id is None else _camera_instances.get(str(camera_id))
    if cam is None:
        raise HardwareError("Camera not initialised")
    return cam.capture()


def camera_capture_all() -> dict[str, np.ndarray]:
    """Capture one frame from every configured camera in acquisition order."""
    if not _camera_instances:
        raise HardwareError("Camera not initialised")
    return {camera_id: cam.capture() for camera_id, cam in _camera_instances.items()}


def camera_set_exposure(
    exposure_us: int,
    gain: Optional[float] = None,
    camera_id: str | None = None,
) -> None:
    """Set camera exposure if the active camera driver supports it."""
    cam = _camera_instance if camera_id is None else _camera_instances.get(str(camera_id))
    if cam is None:
        raise HardwareError("Camera not initialised")
    if not hasattr(cam, "set_exposure"):
        raise HardwareError("Camera exposure control is not supported by this driver")
    cam.set_exposure(exposure_us, gain)


def motor_step(n: int, direction: str = "clockwise") -> None:
    """Move the stepper motor *n* steps in *direction*."""
    if _motor_instance is None:
        raise HardwareError("Motor not initialised")
    _motor_instance.motor_step(n, direction)
    if not _ON_PI and hasattr(_motor_instance, "current_angle_rad"):
        for cam in _camera_instances.values():
            if hasattr(cam, "set_rotation_angle"):
                cam.set_rotation_angle(_motor_instance.current_angle_rad)


def laser_set(state: bool) -> None:
    """Enable or disable the laser."""
    if _laser_instance is None:
        raise HardwareError("Laser not initialised")
    _laser_instance.laser_set(state)


def led_set(color: str, state: bool) -> None:
    """Set an LED on or off."""
    if _led_instance is None:
        raise HardwareError("LED not initialised")
    _led_instance.led_set(color, state)


def led_blink(color: str, frequency_hz: float) -> None:
    """Start blinking an LED at the given frequency."""
    if _led_instance is None:
        raise HardwareError("LED not initialised")
    _led_instance.led_blink(color, frequency_hz)


def display_text(text: str, line: int = 0) -> None:
    """Write *text* on the display at *line*."""
    if _display_instance is None:
        raise HardwareError("Display not initialised")
    _display_instance.display_text(text, line)


def display_status(state: str) -> None:
    """Update the display to show the current scanner *state*."""
    if _display_instance is None:
        raise HardwareError("Display not initialised")
    _display_instance.display_status(state)


__all__ = [
    "HardwareError",
    "init_hardware",
    "camera_capture",
    "camera_capture_all",
    "camera_set_exposure",
    "motor_step",
    "laser_set",
    "led_set",
    "led_blink",
    "display_text",
    "display_status",
]
