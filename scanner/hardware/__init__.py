"""scanner.hardware — Hardware abstraction layer.

Auto-detects whether running on a Raspberry Pi (gpiozero available) and
selects real hardware drivers or mock implementations accordingly.

Exports:
    HardwareError: raised for any hardware-level failure.
    init_hardware: initialise all hardware singletons from config.
    camera_capture: capture a BGR frame from the camera.
    motor_step: advance the stepper motor N steps.
    laser_set: enable / disable the laser.
    led_set: set an LED color on/off.
    led_blink: blink an LED at a given frequency.
    display_text: write text to the display.
    display_status: update display with scanner state.
"""

import copy
import logging
from contextlib import contextmanager
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Custom exception
# --------------------------------------------------------------------------- #


class HardwareError(Exception):
    """Raised when a hardware operation fails or the hardware is unavailable."""


# --------------------------------------------------------------------------- #
# Auto-detect platform
# --------------------------------------------------------------------------- #

_ON_PI: bool = False

try:
    import gpiozero  # noqa: F401

    _ON_PI = True
    logger.info("gpiozero detected — using real hardware drivers")
except ImportError:
    logger.info("gpiozero not found — using mock hardware drivers")

# --------------------------------------------------------------------------- #
# Singleton instances (lazy-initialised via init_hardware)
# --------------------------------------------------------------------------- #

_camera_instance = None
_motor_instance = None
_laser_instance = None
_led_instance = None
_display_instance = None
_camera_config: dict | None = None


def _create_camera_instance(camera_config: dict):
    if _ON_PI:
        from scanner.hardware.camera import PiCamera

        return PiCamera(camera_config)

    from scanner.hardware.mock import MockCamera

    return MockCamera(camera_config)


def init_hardware(config: dict) -> None:
    """Initialise all hardware singletons from *config*.

    Args:
        config: Full settings dict (loaded from settings.yaml).

    Raises:
        HardwareError: if any hardware component fails to initialise.
    """
    global _camera_instance, _motor_instance, _laser_instance
    global _led_instance, _display_instance
    global _camera_config

    logger.info("Initialising hardware (on_pi=%s)", _ON_PI)

    iface_cfg = config.get("interface", {})
    display_type = str(iface_cfg.get("display_type", "oled")).lower()
    display_enabled = display_type not in ("none", "off", "disabled")
    _camera_config = copy.deepcopy(config.get("camera", {}))

    try:
        if _ON_PI:
            from scanner.hardware.motor import StepperMotor
            from scanner.hardware.laser import Laser
            from scanner.hardware.led import LED
            from scanner.hardware.display import Display

            _camera_instance = _create_camera_instance(_camera_config)
            _motor_instance = StepperMotor(config.get("motor", {}))
            _laser_instance = Laser(config.get("laser", {}))
            _led_instance = LED(iface_cfg)
            if display_enabled:
                _display_instance = Display(iface_cfg)
            else:
                _display_instance = None
                logger.info("Display disabled by config (interface.display_type=%s)", display_type)
        else:
            from scanner.hardware.mock import MockCamera, MockMotor, MockLaser, MockLED, MockDisplay

            _camera_instance = _create_camera_instance(_camera_config)
            _motor_instance = MockMotor(config.get("motor", {}))
            _laser_instance = MockLaser(config.get("laser", {}))
            _led_instance = MockLED(iface_cfg)
            if display_enabled:
                _display_instance = MockDisplay(iface_cfg)
            else:
                _display_instance = None
                logger.info("Mock display disabled by config (interface.display_type=%s)", display_type)
    except HardwareError:
        raise
    except Exception as exc:
        raise HardwareError(f"Hardware initialisation failed: {exc}") from exc


def camera_capture() -> np.ndarray:
    """Capture one frame from the camera.

    Returns:
        BGR image as numpy array of shape (H, W, 3), dtype uint8.

    Raises:
        HardwareError: if the camera is not initialised or capture fails.
    """
    if _camera_instance is None:
        raise HardwareError("Camera not initialised — call init_hardware() first")
    return _camera_instance.capture()


def camera_reconfigure(config: dict) -> None:
    """Recreate the camera singleton with a new configuration."""
    global _camera_instance, _camera_config

    if _camera_instance is not None and hasattr(_camera_instance, "close"):
        try:
            _camera_instance.close()
        except Exception as exc:
            logger.warning("Camera close during reconfigure failed: %s", exc)

    try:
        _camera_instance = _create_camera_instance(config)
        _camera_config = copy.deepcopy(config)
    except Exception as exc:
        raise HardwareError(f"Camera reconfiguration failed: {exc}") from exc

    if not _ON_PI and _motor_instance is not None and hasattr(_motor_instance, "current_angle_rad"):
        _camera_instance.set_rotation_angle(_motor_instance.current_angle_rad)


@contextmanager
def camera_temporary_config(overrides: dict):
    """Temporarily apply camera settings for a calibration or debug action."""
    global _camera_config

    if _camera_config is None or _camera_instance is None:
        raise HardwareError("Camera configuration unavailable — call init_hardware() first")

    original = copy.deepcopy(_camera_config)
    temporary = copy.deepcopy(_camera_config)
    temporary.update(overrides)

    if hasattr(_camera_instance, "update_settings"):
        try:
            _camera_instance.update_settings(temporary)
            _camera_config = copy.deepcopy(temporary)
        except Exception:
            try:
                _camera_instance.update_settings(original)
                _camera_config = copy.deepcopy(original)
            except Exception as restore_exc:
                logger.error("Could not restore original camera settings after failure: %s", restore_exc)
            raise
        try:
            yield
        finally:
            _camera_instance.update_settings(original)
            _camera_config = copy.deepcopy(original)
        return

    try:
        camera_reconfigure(temporary)
    except Exception:
        try:
            camera_reconfigure(original)
        except Exception as restore_exc:
            logger.error("Could not restore original camera config after failure: %s", restore_exc)
        raise
    try:
        yield
    finally:
        camera_reconfigure(original)


def motor_step(n: int, direction: str = "clockwise") -> None:
    """Move the stepper motor *n* steps in *direction*.

    Args:
        n: Number of steps (positive integer).
        direction: 'clockwise' or 'counterclockwise'.

    Raises:
        HardwareError: if the motor is not initialised.
    """
    if _motor_instance is None:
        raise HardwareError("Motor not initialised — call init_hardware() first")
    _motor_instance.motor_step(n, direction)
    # Keep MockCamera in sync with MockMotor rotation angle.
    if not _ON_PI and _camera_instance is not None and hasattr(_motor_instance, "current_angle_rad"):
        _camera_instance.set_rotation_angle(_motor_instance.current_angle_rad)


def laser_set(state: bool) -> None:
    """Enable or disable the laser.

    Args:
        state: True to turn the laser on, False to turn it off.

    Raises:
        HardwareError: if the laser is not initialised.
    """
    if _laser_instance is None:
        raise HardwareError("Laser not initialised — call init_hardware() first")
    _laser_instance.laser_set(state)


def led_set(color: str, state: bool) -> None:
    """Set an LED on or off.

    Args:
        color: LED colour string, e.g. 'green', 'orange', 'red'.
        state: True = on, False = off.

    Raises:
        HardwareError: if the LED controller is not initialised.
    """
    if _led_instance is None:
        raise HardwareError("LED not initialised — call init_hardware() first")
    _led_instance.led_set(color, state)


def led_blink(color: str, frequency_hz: float) -> None:
    """Start blinking an LED at the given frequency.

    Args:
        color: LED colour string.
        frequency_hz: Blink frequency in Hz (0 stops blinking).

    Raises:
        HardwareError: if the LED controller is not initialised.
    """
    if _led_instance is None:
        raise HardwareError("LED not initialised — call init_hardware() first")
    _led_instance.led_blink(color, frequency_hz)


def display_text(text: str, line: int = 0) -> None:
    """Write *text* on the display at *line*.

    Args:
        text: String to display.
        line: Line number (0-based).

    Raises:
        HardwareError: if the display is not initialised.
    """
    if _display_instance is None:
        raise HardwareError("Display not initialised — call init_hardware() first")
    _display_instance.display_text(text, line)


def display_status(state: str) -> None:
    """Update the display to show the current scanner *state*.

    Args:
        state: State string, e.g. 'IDLE', 'SCANNING', 'ERROR'.

    Raises:
        HardwareError: if the display is not initialised.
    """
    if _display_instance is None:
        raise HardwareError("Display not initialised — call init_hardware() first")
    _display_instance.display_status(state)


__all__ = [
    "HardwareError",
    "init_hardware",
    "camera_capture",
    "camera_reconfigure",
    "camera_temporary_config",
    "motor_step",
    "laser_set",
    "led_set",
    "led_blink",
    "display_text",
    "display_status",
]
