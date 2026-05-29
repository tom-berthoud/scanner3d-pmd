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


class DoorOpenError(HardwareError):
    """Raised when the safety door interlock trips (door opened) during a scan.

    Subclasses HardwareError so existing error-recovery paths (which turn the
    laser off) handle it correctly.
    """


_ON_PI: bool = False
try:
    import gpiozero  # noqa: F401

    _ON_PI = True
    logger.info("gpiozero detected - using real hardware drivers")
except ImportError:
    logger.info("gpiozero not found - using mock hardware drivers")


_camera_instance = None
_camera_instances: dict[str, object] = {}
_camera_configs: dict[str, dict] = {}
_failed_camera_configs: dict[str, dict] = {}
_motor_instance = None
_laser_instance = None
_led_instance = None
_display_instance = None
_door_instance = None
_hardware_config: dict = {}


def init_hardware(config: dict) -> None:
    """Initialise all hardware singletons from *config*."""
    global _camera_instance, _camera_instances, _camera_configs, _failed_camera_configs
    global _motor_instance, _laser_instance, _led_instance, _display_instance, _hardware_config
    global _door_instance

    from scanner.calibration import camera_configs

    _hardware_config = config
    logger.info("Initialising hardware (on_pi=%s)", _ON_PI)
    iface_cfg = config.get("interface", {})
    door_cfg = config.get("safety", {}).get("door_interlock", {})
    display_type = str(iface_cfg.get("display_type", "oled")).lower()
    display_enabled = display_type not in ("none", "off", "disabled")

    try:
        if _ON_PI:
            from scanner.hardware.camera import PiCamera
            from scanner.hardware.display import Display
            from scanner.hardware.door import DoorSensor
            from scanner.hardware.laser import Laser
            from scanner.hardware.led import LED
            from scanner.hardware.motor import StepperMotor
            from scanner.hardware.usb_camera import USBCamera

            _camera_instances = {}
            _camera_configs = {}
            _failed_camera_configs = {}
            for cam_cfg in camera_configs(config):
                cam_id = str(cam_cfg.get("id", "main"))
                cam_type = str(cam_cfg.get("type", "pi")).lower()
                _camera_configs[cam_id] = cam_cfg
                try:
                    _camera_instances[cam_id] = (
                        USBCamera(cam_cfg) if cam_type == "usb" else PiCamera(cam_cfg)
                    )
                except Exception as exc:
                    _failed_camera_configs[cam_id] = cam_cfg
                    logger.warning("Camera %s init failed: %s", cam_id, exc)
            _motor_instance = StepperMotor(config.get("motor", {}))
            _laser_instance = Laser(config.get("laser", {}))
            _led_instance = LED(iface_cfg)
            _display_instance = Display(iface_cfg) if display_enabled else None
            _door_instance = DoorSensor(door_cfg)
        else:
            from scanner.hardware.mock import (
                MockCamera,
                MockDisplay,
                MockDoorSensor,
                MockLED,
                MockLaser,
                MockMotor,
            )

            _camera_configs = {
                str(cam_cfg.get("id", "main")): cam_cfg for cam_cfg in camera_configs(config)
            }
            _failed_camera_configs = {}
            _camera_instances = {
                camera_id: MockCamera(cam_cfg) for camera_id, cam_cfg in _camera_configs.items()
            }
            _motor_instance = MockMotor(config.get("motor", {}))
            _laser_instance = MockLaser(config.get("laser", {}))
            _led_instance = MockLED(iface_cfg)
            _display_instance = MockDisplay(iface_cfg) if display_enabled else None
            _door_instance = MockDoorSensor(door_cfg)

        if not _camera_instances:
            raise HardwareError("No camera could be initialised")
        else:
            _camera_instance = next(iter(_camera_instances.values()))
        if _display_instance is None:
            logger.info("Display disabled by config (interface.display_type=%s)", display_type)
    except HardwareError:
        raise
    except Exception as exc:
        raise HardwareError(f"Hardware initialisation failed: {exc}") from exc


def _ensure_camera(camera_id: str | None = None) -> object | None:
    global _camera_instance

    cam = _camera_instance if camera_id is None else _camera_instances.get(str(camera_id))
    if cam is not None:
        return cam
    if not _ON_PI or camera_id is None:
        return None
    cam_cfg = _failed_camera_configs.get(str(camera_id)) or _camera_configs.get(str(camera_id))
    if not cam_cfg:
        return None
    try:
        from scanner.hardware.camera import PiCamera
        from scanner.hardware.usb_camera import USBCamera

        cam_type = str(cam_cfg.get("type", "pi")).lower()
        cam = USBCamera(cam_cfg) if cam_type == "usb" else PiCamera(cam_cfg)
        _camera_instances[str(camera_id)] = cam
        _failed_camera_configs.pop(str(camera_id), None)
        if _camera_instance is None:
            _camera_instance = cam
        logger.info("Camera %s initialised after retry", camera_id)
        return cam
    except Exception as exc:
        _failed_camera_configs[str(camera_id)] = cam_cfg
        logger.warning("Camera %s retry init failed: %s", camera_id, exc)
        return None


def camera_capture(camera_id: str | None = None) -> np.ndarray:
    """Capture one frame from one camera.

    If *camera_id* is omitted, the first configured camera is used for legacy
    callers.
    """
    cam = _ensure_camera(camera_id)
    if cam is None:
        raise HardwareError("Camera not initialised")
    return cam.capture()


def camera_capture_all() -> dict[str, np.ndarray]:
    """Capture one frame from every configured camera in acquisition order."""
    frames: dict[str, np.ndarray] = {}
    for camera_id in _camera_configs:
        cam = _ensure_camera(camera_id)
        if cam is not None:
            frames[camera_id] = cam.capture()
    if not frames:
        raise HardwareError("Camera not initialised")
    return frames


def camera_set_exposure(
    exposure_us: int,
    gain: Optional[float] = None,
    camera_id: str | None = None,
) -> None:
    """Set camera exposure if the active camera driver supports it."""
    cam = _ensure_camera(camera_id)
    if cam is None:
        raise HardwareError("Camera not initialised")
    if not hasattr(cam, "set_exposure"):
        raise HardwareError("Camera exposure control is not supported by this driver")
    cam.set_exposure(exposure_us, gain)


def camera_set_controls(camera_id: str, controls: dict) -> dict:
    """Apply camera controls such as resolution, exposure, gain and focus."""
    cam = _ensure_camera(camera_id)
    if cam is None:
        raise HardwareError(f"Camera not initialised: {camera_id}")
    if not hasattr(cam, "set_controls"):
        raise HardwareError("Camera control update is not supported by this driver")
    return cam.set_controls(controls)


def camera_get_info(camera_id: str | None = None) -> dict:
    """Return requested and driver-reported camera settings."""
    cam = _ensure_camera(camera_id)
    if cam is None:
        raise HardwareError("Camera not initialised")
    if hasattr(cam, "get_info"):
        return cam.get_info()
    return {"camera_id": camera_id, "driver": type(cam).__name__}


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


def door_interlock_enabled() -> bool:
    """Return True if the safety door interlock is active."""
    return bool(_door_instance is not None and _door_instance.enabled)


def door_is_open() -> bool:
    """Return True if the door is open *and* the interlock is enabled.

    Returns False when the interlock is disabled or no sensor is present, so
    callers never block when the feature is turned off.
    """
    if _door_instance is None:
        return False
    return bool(_door_instance.is_open())


def check_door_interlock() -> None:
    """Raise :class:`DoorOpenError` if the door is open while the interlock is on.

    No-op when the interlock is disabled. Call this before energising the
    laser so an open door aborts the operation immediately.
    """
    if door_is_open():
        raise DoorOpenError("Safety door is open — operation aborted")


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
    "DoorOpenError",
    "init_hardware",
    "camera_capture",
    "camera_capture_all",
    "camera_get_info",
    "camera_set_controls",
    "camera_set_exposure",
    "motor_step",
    "laser_set",
    "door_interlock_enabled",
    "door_is_open",
    "check_door_interlock",
    "led_set",
    "led_blink",
    "display_text",
    "display_status",
]
