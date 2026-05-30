"""scanner.hardware.mock — Synthetic hardware for development without a Raspberry Pi.

Provides MockCamera, MockMotor, MockLaser, MockLED, MockDisplay and
MockDoorSensor that mimic the real hardware API without requiring any GPIO or
camera hardware.
"""

import logging
import math
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Mock geometry constants
# --------------------------------------------------------------------------- #

_CUBE_HALF_MM = 35.0        # virtual cube half-side length

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480

# Laser / camera geometry — must match config/laser_plane.yaml and
# config/camera_intrinsics.yaml (reference resolution 640×480).
_LASER_A = 0.5
_LASER_C = 0.866
_LASER_D = -259.8   # plane: 0.5·x + 0.866·z = 259.8
_TURNTABLE_Z = 300.0  # object centre Z in camera frame (mm)
_REF_W = 640.0
_REF_H = 480.0
_REF_FX = 800.0


class MockCamera:
    """Simulated Pi Camera Module 3.

    Generates synthetic BGR images with a geometrically correct green laser
    line on a virtual **cube** placed on a turntable. The geometry is fully
    consistent with the triangulation pipeline.

    Args:
        config: Camera configuration dict. The ``mock_shape`` key is accepted
            for API parity but the mock always renders a cube.
    """

    def __init__(self, config: dict) -> None:
        res = config.get("resolution", [_IMAGE_WIDTH, _IMAGE_HEIGHT])
        self._width: int = int(res[0])
        self._height: int = int(res[1])
        self._exposure_us: int = int(config.get("exposure_us", 5000))
        self._gain: float = float(config.get("gain", 1.0))
        # Only the cube is supported; kept for API parity / get_info reporting.
        self._shape: str = "cube"
        # Shared angle reference — updated by MockMotor via hardware.__init__
        self._rotation_angle_rad: float = 0.0
        logger.debug(
            "MockCamera initialised (%dx%d, shape=cube, exposure=%d µs)",
            self._width,
            self._height,
            self._exposure_us,
        )

    def set_rotation_angle(self, angle_rad: float) -> None:
        """Set the current rotation angle used for laser line simulation.

        Args:
            angle_rad: Rotation angle in radians.
        """
        self._rotation_angle_rad = angle_rad

    def set_exposure(self, exposure_us: int, gain: Optional[float] = None) -> None:
        """Store exposure settings for API parity with the real camera."""
        self._exposure_us = int(exposure_us)
        if gain is not None:
            self._gain = float(gain)
        logger.debug("MockCamera exposure set to %d us, gain=%.2f", self._exposure_us, self._gain)

    def set_controls(self, controls: dict) -> dict:
        """Store runtime camera controls for API parity with real drivers."""
        if controls.get("width") is not None and controls.get("height") is not None:
            self._width = int(controls["width"])
            self._height = int(controls["height"])
        if controls.get("exposure_us") is not None:
            self._exposure_us = int(controls["exposure_us"])
        if controls.get("gain") is not None:
            self._gain = float(controls["gain"])
        return self.get_info()

    def get_info(self) -> dict:
        """Return requested mock camera settings."""
        return {
            "driver": "MockCamera",
            "requested": {
                "width": self._width,
                "height": self._height,
                "exposure_us": self._exposure_us,
                "gain": self._gain,
                "mock_shape": self._shape,
            },
            "actual": {
                "width": self._width,
                "height": self._height,
                "exposure_us": self._exposure_us,
                "gain": self._gain,
            },
        }

    def capture(self) -> np.ndarray:
        """Generate a synthetic BGR image with a geometrically correct laser line.

        For each image column, casts a ray through the laser plane to find the
        3-D surface point of the virtual cube, then projects the bright green
        laser reflection onto the image. The geometry is fully consistent with
        ``triangulate()``: accumulated over the rotation steps it reconstructs
        the cube in world space. The cube rotates on the turntable, so the
        visible laser profile changes with ``_rotation_angle_rad``.

        Returns:
            BGR image of shape (H, W, 3), dtype uint8.
        """
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        noise = np.random.randint(0, 8, (self._height, self._width, 3), dtype=np.uint8)
        frame += noise

        # Camera intrinsics scaled to actual resolution (reference: 800 @ 640×480)
        fx = _REF_FX * (self._width / _REF_W)
        fy = _REF_FX * (self._height / _REF_H)
        img_cx = self._width / 2.0
        img_cy = self._height / 2.0

        theta = self._rotation_angle_rad  # current turntable angle

        for u_px in range(self._width):
            x_n = (u_px - img_cx) / fx
            # t = depth to laser plane (b=0 ⟹ independent of y_n)
            denom = _LASER_A * x_n + _LASER_C
            if abs(denom) < 1e-9:
                continue
            t = -_LASER_D / denom
            if t <= 0.0:
                continue
            x_cam = t * x_n
            z_cam = t

            y_cam = self._surface_y(x_cam, z_cam, theta)
            if y_cam is None:
                continue

            v_f = fy * (y_cam / z_cam) + img_cy
            row = int(round(v_f))
            if 0 <= row < self._height:
                for dr in range(-3, 4):
                    r = row + dr
                    if 0 <= r < self._height:
                        intensity = int(220 * math.exp(-0.5 * (dr / 1.5) ** 2))
                        frame[r, u_px, 1] = min(255, intensity)

        logger.debug(
            "MockCamera.capture() → frame %dx%d shape=cube angle=%.3f rad",
            self._width,
            self._height,
            theta,
        )
        return frame

    def _surface_y(self, x_cam: float, z_cam: float, theta: float) -> Optional[float]:
        """Return the upper-surface Y coordinate (mm) of the cube at a laser-plane point.

        The point (x_cam, ?, z_cam) lies on the laser plane. This method finds
        the Y value of the cube surface at that (X, Z) position in world
        coordinates (after un-rotating the turntable by *theta*).

        Args:
            x_cam: X coordinate of the laser-plane point in camera frame (mm).
            z_cam: Z coordinate of the laser-plane point in camera frame (mm).
            theta: Current turntable rotation angle (radians).

        Returns:
            Y coordinate (mm, top face) or ``None`` if the laser ray misses the
            cube at this column.
        """
        # Un-rotate from camera frame to object/world frame.
        # P_world = R(−θ) @ (P_cam − T)  where T = [0, 0, _TURNTABLE_Z]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = x_cam
        dz = z_cam - _TURNTABLE_Z
        x_w = cos_t * dx + sin_t * dz
        z_w = -sin_t * dx + cos_t * dz

        # Cube (half-side H, centred at world origin, aligned with axes).
        # Top face at y = H; sides appear when the laser crosses an edge.
        H = _CUBE_HALF_MM
        if abs(x_w) > H or abs(z_w) > H:
            return None  # outside cube footprint
        return H


class MockMotor:
    """Simulated NEMA 17 stepper motor with STEP/DIR driver.

    Maintains the current absolute step count and computes the rotation angle
    in radians.  Provides the angle to MockCamera when both are used together.

    Args:
        config: Motor configuration dict (steps_per_rev, accel_steps …).
    """

    def __init__(self, config: dict) -> None:
        self._steps_per_rev: int = int(config.get("steps_per_rev", 200))
        self._microstepping: int = int(config.get("microstepping", 1))
        self._accel_steps: int = int(config.get("accel_steps", 20))
        self._total_steps: int = self._steps_per_rev * self._microstepping
        self._current_step: int = 0
        logger.debug(
            "MockMotor initialised (steps_per_rev=%d, microstepping=%d)",
            self._steps_per_rev,
            self._microstepping,
        )

    @property
    def current_angle_rad(self) -> float:
        """Current rotation angle in radians."""
        return 2.0 * math.pi * self._current_step / self._total_steps

    def motor_step(self, n: int, direction: str = "clockwise") -> None:
        """Simulate advancing *n* steps in *direction*.

        Args:
            n: Number of steps (positive integer).
            direction: 'clockwise' or 'counterclockwise'.

        Raises:
            ValueError: if *direction* is not recognised.
        """
        if direction not in ("clockwise", "counterclockwise"):
            raise ValueError(f"Unknown direction: {direction!r}")
        delta = n if direction == "clockwise" else -n
        self._current_step = (self._current_step + delta) % self._total_steps
        logger.debug(
            "MockMotor.motor_step(n=%d, dir=%s) → step=%d, angle=%.3f rad",
            n,
            direction,
            self._current_step,
            self.current_angle_rad,
        )

    def reset(self) -> None:
        """Reset step counter to zero (home position)."""
        self._current_step = 0


class MockLaser:
    """Simulated laser line module.

    Maintains an on/off state.  The laser is OFF by default at startup
    (safety rule, section 8 of agents.md).

    Args:
        config: Laser configuration dict (warmup_ms …).
    """

    def __init__(self, config: dict) -> None:
        self._warmup_ms: int = int(config.get("warmup_ms", 50))
        self._state: bool = False  # OFF by default — non-negotiable safety rule
        logger.debug("MockLaser initialised (state=OFF, warmup=%d ms)", self._warmup_ms)

    @property
    def state(self) -> bool:
        """Current laser state (True = on)."""
        return self._state

    def laser_set(self, state: bool) -> None:
        """Set laser on or off.

        Args:
            state: True to turn on, False to turn off.
        """
        if state and not self._state:
            # Simulate warmup delay
            time.sleep(self._warmup_ms / 1000.0)
        self._state = state
        logger.debug("MockLaser → %s", "ON" if state else "OFF")


class MockLED:
    """Simulated RGB LED controller.

    Args:
        config: Interface configuration dict (unused, kept for API symmetry).
    """

    def __init__(self, config: dict) -> None:
        self._states: dict[str, bool] = {}
        self._blink_freqs: dict[str, float] = {}
        logger.debug("MockLED initialised")

    def led_set(self, color: str, state: bool) -> None:
        """Set LED *color* on or off.

        Args:
            color: LED colour string ('green', 'orange', 'red', …).
            state: True = on, False = off.
        """
        self._states[color] = state
        # Stop any blinking for this colour
        self._blink_freqs.pop(color, None)
        logger.debug("MockLED.led_set(%s, %s)", color, state)

    def led_blink(self, color: str, frequency_hz: float) -> None:
        """Start blinking LED *color* at *frequency_hz* Hz.

        A frequency of 0 stops blinking and turns the LED off.

        Args:
            color: LED colour string.
            frequency_hz: Blink frequency in Hz.
        """
        if frequency_hz <= 0:
            self._blink_freqs.pop(color, None)
            self._states[color] = False
        else:
            self._blink_freqs[color] = frequency_hz
            self._states[color] = True
        logger.debug("MockLED.led_blink(%s, %.2f Hz)", color, frequency_hz)

    def get_state(self, color: str) -> bool:
        """Return the current on/off state of *color* LED.

        Args:
            color: LED colour string.

        Returns:
            True if the LED is on (or blinking), False otherwise.
        """
        return self._states.get(color, False)


class MockDisplay:
    """Simulated TFT SPI display (RB-TFT3.2-V2).

    Logs display updates instead of driving real SPI hardware.

    Args:
        config: Interface configuration dict (unused, kept for API symmetry).
    """

    _MAX_LINES = 8

    def __init__(self, config: dict) -> None:
        self._lines: list[str] = [""] * self._MAX_LINES
        logger.debug("MockDisplay initialised (simulating RB-TFT3.2-V2)")

    def display_text(self, text: str, line: int = 0) -> None:
        """Write *text* on display line *line*.

        Args:
            text: String to display (truncated to 32 characters).
            line: Line index (0-based, clamped to valid range).
        """
        line = max(0, min(line, self._MAX_LINES - 1))
        self._lines[line] = text[:32]
        logger.info("MockDisplay [line %d]: %s", line, text[:32])

    def display_status(self, state: str) -> None:
        """Display the scanner state on line 0.

        Args:
            state: State string, e.g. 'IDLE', 'SCANNING'.
        """
        self.display_text(f"State: {state}", line=0)


class MockDoorSensor:
    """Simulated safety door interlock.

    Mirrors :class:`scanner.hardware.door.DoorSensor` without any GPIO. The
    simulated door defaults to *closed*; tests (or a dev UI) can flip it with
    :meth:`set_open`. When the interlock is disabled via config, :meth:`is_open`
    always returns False.

    Args:
        config: ``safety.door_interlock`` configuration dict (only ``enabled``
            is honoured here).
    """

    def __init__(self, config: dict) -> None:
        config = config or {}
        self._enabled: bool = bool(config.get("enabled", False))
        self._open: bool = False
        logger.debug("MockDoorSensor initialised (enabled=%s, state=closed)", self._enabled)

    @property
    def enabled(self) -> bool:
        """True when the interlock is active."""
        return self._enabled

    def is_open(self) -> bool:
        """Return True when the door is open and the interlock is enabled."""
        return self._enabled and self._open

    def set_open(self, is_open: bool) -> None:
        """Simulate the door opening or closing (development/testing helper)."""
        self._open = bool(is_open)
        logger.debug("MockDoorSensor → %s", "OPEN" if self._open else "CLOSED")

    def close(self) -> None:
        """No-op for API symmetry with the real driver."""
