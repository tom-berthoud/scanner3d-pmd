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

_SPHERE_RADIUS_MM = 40.0   # virtual sphere radius
_CYLINDER_RADIUS_MM = 35.0  # virtual cylinder radius
_CYLINDER_HEIGHT_MM = 70.0  # virtual cylinder height (centred on turntable)
_CUBE_HALF_MM = 35.0        # virtual cube half-side length

# Mushroom geometry (all mm, world frame — Y = up)
_MUSH_CAP_R    = 30.0   # cap hemisphere radius
_MUSH_CAP_CY   = 10.0   # cap centre Y (flat face of hemisphere)
_MUSH_STEM_R   =  9.0   # stem cylinder radius
_MUSH_STEM_BOT = -38.0  # stem bottom Y
_MUSH_STEM_TOP =  10.0  # stem top Y (= cap centre)

# Duck geometry (all mm, world frame — Y = up)
# Body: oblate ellipsoid centred near origin
_DUCK_BODY_RX  = 26.0   # body half-width  (left-right)
_DUCK_BODY_RY  = 19.0   # body half-height (up-down)
_DUCK_BODY_RZ  = 22.0   # body half-depth  (front-back)
_DUCK_BODY_CY  = -3.0   # body centre Y (slightly below turntable mid)
_DUCK_BODY_CZ  =  0.0   # body centre Z in world (on rotation axis)
# Head: sphere sitting on top of / slightly in front of body
_DUCK_HEAD_R   = 13.0   # head radius
_DUCK_HEAD_CY  = 27.0   # head centre Y
_DUCK_HEAD_CZ  = -14.0  # head centre Z (forward of body in initial orientation)
# Neck: narrow ellipsoid connecting body to head
_DUCK_NECK_RX  =  7.0
_DUCK_NECK_RY  = 10.0
_DUCK_NECK_RZ  =  7.0
_DUCK_NECK_CY  = 12.0   # midpoint between body top and head bottom
_DUCK_NECK_CZ  = -8.0
# Bill: flat ellipsoid protruding forward from head
_DUCK_BILL_RX  =  5.0
_DUCK_BILL_RY  =  2.5
_DUCK_BILL_RZ  =  9.0
_DUCK_BILL_CY  = 22.0
_DUCK_BILL_CZ  = -26.0  # well forward of head centre
# "SCAN" text geometry — 5×7 bitmap font, each pixel = 2.5 mm
# Letters defined as 7 rows of 5 bits (top to bottom), 1 = filled
_SCAN_GLYPHS: dict[str, list[int]] = {
    "S": [
        0b01110,
        0b10001,
        0b10000,
        0b01110,
        0b00001,
        0b10001,
        0b01110,
    ],
    "C": [
        0b01110,
        0b10001,
        0b10000,
        0b10000,
        0b10000,
        0b10001,
        0b01110,
    ],
    "A": [
        0b01110,
        0b10001,
        0b10001,
        0b11111,
        0b10001,
        0b10001,
        0b10001,
    ],
    "N": [
        0b10001,
        0b11001,
        0b10101,
        0b10011,
        0b10001,
        0b10001,
        0b10001,
    ],
}
_SCAN_PX_MM = 3.5       # size of one bitmap pixel in mm
_SCAN_LETTER_W = 5      # pixels per letter width
_SCAN_LETTER_H = 7      # pixels per letter height
_SCAN_GAP_PX = 2        # gap between letters in pixels
_SCAN_TEXT_BUMP = 8.0    # how much taller letter pixels are vs cylinder top (mm)
_SCAN_CYL_R = 38.0      # cylinder base radius (mm) — fits the text width
_SCAN_CYL_H = 30.0      # cylinder half-height (mm)

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480

# Laser / camera geometry — must match config/laser_plane.yaml and
# config/camera_intrinsics.yaml (reference resolution 640×480).
_LASER_A = 0.5
_LASER_C = 0.866
_LASER_D = -259.8   # plane: 0.5·x + 0.866·z = 259.8
_TURNTABLE_Z = 300.0  # sphere / object centre Z in camera frame (mm)
_REF_W = 640.0
_REF_H = 480.0
_REF_FX = 800.0


class MockCamera:
    """Simulated Pi Camera Module 3.

    Generates synthetic BGR images with a geometrically correct green laser
    line on a virtual object (sphere, cylinder or cube) placed on a turntable.
    The geometry is fully consistent with the triangulation pipeline.

    Args:
        config: Camera configuration dict.  Recognised keys beyond the
            standard camera fields:

            ``mock_shape`` (str): object shape — ``'sphere'`` (default),
            ``'cylinder'``, ``'cube'``, ``'duck'``, ``'mushroom'``,
            or ``'scan_text'`` (the word SCAN in relief, readable from above).
    """

    def __init__(self, config: dict) -> None:
        res = config.get("resolution", [_IMAGE_WIDTH, _IMAGE_HEIGHT])
        self._width: int = int(res[0])
        self._height: int = int(res[1])
        self._exposure_us: int = int(config.get("exposure_us", 5000))
        self._gain: float = float(config.get("gain", 1.0))
        self._shape: str = str(config.get("mock_shape", "sphere")).lower()
        # Shared angle reference — updated by MockMotor via hardware.__init__
        self._rotation_angle_rad: float = 0.0
        logger.debug(
            "MockCamera initialised (%dx%d, shape=%s, exposure=%d µs)",
            self._width,
            self._height,
            self._shape,
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
        3-D surface point of the virtual object, then projects the bright green
        laser reflection onto the image.  The geometry is fully consistent with
        ``triangulate()``: accumulated over 200 rotation steps it reconstructs
        the object in world space.

        Supported shapes (``mock_shape`` config key):
            ``'sphere'``   — 40 mm radius sphere, symmetric, angle-independent.
            ``'cylinder'`` — 35 mm radius × 70 mm height cylinder; the laser
                             line follows a sinusoidal height profile as the
                             turntable rotates.
            ``'cube'``     — 35 mm half-side cube; flat faces produce flat line
                             segments with sharp jumps at the edges.

        For cylinder and cube the object rotates on the turntable, so the
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
            "MockCamera.capture() → frame %dx%d shape=%s angle=%.3f rad",
            self._width,
            self._height,
            self._shape,
            theta,
        )
        return frame

    def _surface_y(self, x_cam: float, z_cam: float, theta: float) -> Optional[float]:
        """Return the upper-surface Y coordinate (mm) at the given laser-plane point.

        The point (x_cam, ?, z_cam) lies on the laser plane.  This method
        finds the Y value of the object surface at that (X, Z) position in
        world coordinates (after un-rotating the turntable by *theta*).

        Args:
            x_cam: X coordinate of the laser-plane point in camera frame (mm).
            z_cam: Z coordinate of the laser-plane point in camera frame (mm).
            theta: Current turntable rotation angle (radians).

        Returns:
            Y coordinate (mm, upper surface only, ≥ 0) or ``None`` if the
            laser ray misses the object at this column.
        """
        # Un-rotate from camera frame to object/world frame.
        # P_world = R(−θ) @ (P_cam − T)  where T = [0, 0, _TURNTABLE_Z]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = x_cam
        dz = z_cam - _TURNTABLE_Z
        x_w = cos_t * dx + sin_t * dz
        z_w = -sin_t * dx + cos_t * dz

        # ------------------------------------------------------------------ #
        # Sphere  (radius R, centre at world origin)
        # ------------------------------------------------------------------ #
        if self._shape == "sphere":
            y_sq = _SPHERE_RADIUS_MM**2 - x_w**2 - z_w**2
            if y_sq < 0.0:
                return None
            return math.sqrt(y_sq)  # upper hemisphere

        # ------------------------------------------------------------------ #
        # Cylinder  (radius R, axis = Y axis, height ±H/2)
        # ------------------------------------------------------------------ #
        if self._shape == "cylinder":
            r_sq = _CYLINDER_RADIUS_MM**2 - x_w**2
            if r_sq < 0.0:
                return None
            # Top cap if laser hits above the cylinder half-height, else side
            half_h = _CYLINDER_HEIGHT_MM / 2.0
            y_side = math.sqrt(r_sq)  # always positive (upper half of side)
            if y_side > half_h:
                # Ray misses the cylinder body; check top cap at y = half_h
                if abs(x_w) <= _CYLINDER_RADIUS_MM and abs(z_w) <= _CYLINDER_RADIUS_MM:
                    return half_h
                return None
            return y_side

        # ------------------------------------------------------------------ #
        # Cube  (half-side H, centred at world origin, aligned with axes)
        # ------------------------------------------------------------------ #
        if self._shape == "cube":
            H = _CUBE_HALF_MM
            if abs(x_w) > H or abs(z_w) > H:
                return None  # outside cube footprint
            # Top face at y = H; sides visible when laser crosses an edge
            return H

        # ------------------------------------------------------------------ #
        # Mushroom  (hemisphere cap + cylinder stem)
        # ------------------------------------------------------------------ #
        if self._shape == "mushroom":
            r_xz = math.sqrt(x_w ** 2 + z_w ** 2)
            # Cap: upper hemisphere
            if r_xz <= _MUSH_CAP_R:
                return _MUSH_CAP_CY + math.sqrt(_MUSH_CAP_R ** 2 - r_xz ** 2)
            # Stem: narrow cylinder outside cap footprint not reachable,
            # but capture stem top edge for points exactly at stem radius
            if r_xz <= _MUSH_STEM_R:
                return _MUSH_STEM_TOP
            return None

        # ------------------------------------------------------------------ #
        # Duck  (body ellipsoid + neck ellipsoid + head sphere + bill ellipsoid)
        # ------------------------------------------------------------------ #
        if self._shape == "duck":
            candidates: list[float] = []

            # Helper: upper surface of an axis-aligned ellipsoid
            def _ellipsoid_y(cx: float, cy: float, cz: float,
                             rx: float, ry: float, rz: float) -> Optional[float]:
                xz_val = ((x_w - cx) / rx) ** 2 + ((z_w - cz) / rz) ** 2
                if xz_val > 1.0:
                    return None
                return cy + ry * math.sqrt(1.0 - xz_val)

            # Body
            y = _ellipsoid_y(0.0, _DUCK_BODY_CY, _DUCK_BODY_CZ,
                             _DUCK_BODY_RX, _DUCK_BODY_RY, _DUCK_BODY_RZ)
            if y is not None:
                candidates.append(y)

            # Neck
            y = _ellipsoid_y(0.0, _DUCK_NECK_CY, _DUCK_NECK_CZ,
                             _DUCK_NECK_RX, _DUCK_NECK_RY, _DUCK_NECK_RZ)
            if y is not None:
                candidates.append(y)

            # Head (sphere = ellipsoid with rx=ry=rz)
            y = _ellipsoid_y(0.0, _DUCK_HEAD_CY, _DUCK_HEAD_CZ,
                             _DUCK_HEAD_R, _DUCK_HEAD_R, _DUCK_HEAD_R)
            if y is not None:
                candidates.append(y)

            # Bill
            y = _ellipsoid_y(0.0, _DUCK_BILL_CY, _DUCK_BILL_CZ,
                             _DUCK_BILL_RX, _DUCK_BILL_RY, _DUCK_BILL_RZ)
            if y is not None:
                candidates.append(y)

            return max(candidates) if candidates else None

        # ------------------------------------------------------------------ #
        # SCAN text  (cylinder with "SCAN" embossed on top, readable from Y+)
        # ------------------------------------------------------------------ #
        if self._shape == "scan_text":
            r_xz = math.sqrt(x_w ** 2 + z_w ** 2)

            # Outside cylinder → no surface
            if r_xz > _SCAN_CYL_R:
                return None

            # Cylinder side wall (lower half visible when laser hits far from centre)
            half_h = _SCAN_CYL_H

            # Top surface: check if point falls on a letter pixel
            word = "SCAN"
            n_letters = len(word)
            total_px_w = n_letters * _SCAN_LETTER_W + (n_letters - 1) * _SCAN_GAP_PX
            total_px_h = _SCAN_LETTER_H
            total_w_mm = total_px_w * _SCAN_PX_MM
            total_h_mm = total_px_h * _SCAN_PX_MM

            # Map x_w, z_w to pixel coordinates in the text bitmap
            # x_w → column (left to right), z_w → row (top to bottom, -z = top)
            px_col = (x_w + total_w_mm / 2.0) / _SCAN_PX_MM
            px_row = (-z_w + total_h_mm / 2.0) / _SCAN_PX_MM

            col_i = int(px_col)
            row_i = int(px_row)

            is_letter = False
            if 0 <= row_i < _SCAN_LETTER_H and 0 <= col_i < total_px_w:
                letter_stride = _SCAN_LETTER_W + _SCAN_GAP_PX
                letter_idx = col_i // letter_stride
                local_col = col_i % letter_stride

                if letter_idx < n_letters and local_col < _SCAN_LETTER_W:
                    glyph = _SCAN_GLYPHS.get(word[letter_idx])
                    if glyph and (glyph[row_i] >> (_SCAN_LETTER_W - 1 - local_col)) & 1:
                        is_letter = True

            if is_letter:
                return half_h + _SCAN_TEXT_BUMP  # raised letter surface
            return half_h  # flat cylinder top

        return None


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
