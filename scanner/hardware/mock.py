"""scanner.hardware.mock — Synthetic hardware for development without a Raspberry Pi.

Provides MockCamera, MockMotor, MockLaser, MockDisplay and
MockDoorSensor that mimic the real hardware API without requiring any GPIO or
camera hardware.
"""

import logging
import math
import os
import time
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Mock geometry constants
# --------------------------------------------------------------------------- #

_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480

# Half-thickness (mm) of the laser sheet used to select illuminated points.
_LASER_SLICE_MM = 0.7
# Virtual cube standing on the turntable, centred on the rotation axis.
_CUBE_HALF_MM = 30.0      # half side length
_CUBE_HEIGHT_MM = 60.0    # base at y=0, top face at y=60
_CUBE_STEP_MM = 0.8       # surface sampling resolution


def _build_surface() -> np.ndarray:
    """Tessellate a cube standing on the turntable as an ``(N, 3)`` array.

    The object frame is the turntable-fixed frame used by ``triangulate`` after
    un-rotation: ``+Y`` is the vertical rotation axis. The cube base sits at
    ``y = 0`` and is centred on the axis. The four vertical faces and the top
    face are sampled (the bottom rests on the turntable and is never seen).
    """
    h = _CUBE_HALF_MM
    top = _CUBE_HEIGHT_MM
    n = max(2, int(round((2.0 * h) / _CUBE_STEP_MM)) + 1)
    ny = max(2, int(round(top / _CUBE_STEP_MM)) + 1)
    u = np.linspace(-h, h, n)
    yv = np.linspace(0.0, top, ny)

    parts = []
    # Top face (y = top).
    gx, gz = np.meshgrid(u, u)
    parts.append(np.stack([gx.ravel(), np.full(gx.size, top), gz.ravel()], axis=1))
    # Four vertical faces.
    gy, gt = np.meshgrid(yv, u)
    gy = gy.ravel()
    gt = gt.ravel()
    parts.append(np.stack([np.full(gy.size, h), gy, gt], axis=1))    # x = +h
    parts.append(np.stack([np.full(gy.size, -h), gy, gt], axis=1))   # x = -h
    parts.append(np.stack([gt, gy, np.full(gy.size, h)], axis=1))    # z = +h
    parts.append(np.stack([gt, gy, np.full(gy.size, -h)], axis=1))   # z = -h
    return np.vstack(parts).astype(np.float64)


class MockCamera:
    """Simulated camera with a geometrically exact synthetic laser line.

    Renders the green laser line as the **forward projection** of a virtual
    cube: for the current turntable angle, the points of the cube that
    lie on the laser plane are projected through the camera's real calibration
    (intrinsics + distortion + extrinsics). This is the exact inverse of
    :func:`scanner.processing.triangulation.triangulate`, so accumulating the
    frames over a full rotation reconstructs the piece in the platform frame —
    and the two configured cameras agree on a single shared object.

    When no usable per-camera calibration is available (e.g. a bare
    ``MockCamera(config)`` without ``camera_id``), it falls back to a simple
    synthetic frame so the API never breaks.

    Args:
        config: Camera configuration dict (resolution, exposure …).
        camera_id: Camera id used to load the matching calibration.
        full_config: Full settings dict, required to load the camera model.
    """

    def __init__(
        self,
        config: dict,
        camera_id: Optional[str] = None,
        full_config: Optional[dict] = None,
    ) -> None:
        res = config.get("resolution", [_IMAGE_WIDTH, _IMAGE_HEIGHT])
        self._width: int = int(res[0])
        self._height: int = int(res[1])
        self._exposure_us: int = int(config.get("exposure_us", 5000))
        self._gain: float = float(config.get("gain", 1.0))
        self._shape: str = "cube"
        self._camera_id: Optional[str] = str(camera_id) if camera_id is not None else None
        self._full_config: Optional[dict] = full_config
        self._rotation_angle_rad: float = 0.0

        # Lazily-loaded geometric model (calibration + virtual surface).
        self._model_loaded: bool = False
        self._geometric: bool = False
        self._K: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None
        self._t: Optional[np.ndarray] = None
        self._plane: Optional[np.ndarray] = None
        self._axis: np.ndarray = np.zeros(3, dtype=np.float64)
        self._surface: Optional[np.ndarray] = None

        logger.debug(
            "MockCamera initialised (%dx%d, id=%s, exposure=%d µs)",
            self._width,
            self._height,
            self._camera_id,
            self._exposure_us,
        )

    # ------------------------------------------------------------------ API
    def set_rotation_angle(self, angle_rad: float) -> None:
        """Set the current turntable rotation angle used for the laser profile."""
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

    # ------------------------------------------------------------- rendering
    def _ensure_model(self) -> None:
        """Load this camera's calibration and the virtual surface (once)."""
        if self._model_loaded:
            return
        self._model_loaded = True
        if self._full_config is None or self._camera_id is None:
            logger.info(
                "MockCamera %s: no calibration context, using flat fallback render",
                self._camera_id,
            )
            return
        try:
            from scanner.calibration import load_camera_model

            K, dist, plane, R, t = load_camera_model(self._full_config, self._camera_id)
            self._K = np.asarray(K, dtype=np.float64)
            self._dist = np.asarray(dist, dtype=np.float64).reshape(-1)
            self._R = np.asarray(R, dtype=np.float64)
            self._t = np.asarray(t, dtype=np.float64).reshape(3)
            self._plane = np.asarray(plane, dtype=np.float64).reshape(4)
            self._axis = self._load_axis_point()
            self._surface = _build_surface()
            self._geometric = True
            logger.info("MockCamera %s: geometric render enabled", self._camera_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "MockCamera %s: calibration unavailable (%s), using flat fallback",
                self._camera_id,
                exc,
            )

    def _load_axis_point(self) -> np.ndarray:
        """Read the turntable rotation-axis point from config/platform.yaml."""
        here = os.path.dirname(__file__)
        path = os.path.join(here, "..", "..", "config", "platform.yaml")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                ap = data.get("rotation_axis_point_mm")
                if ap is not None:
                    return np.asarray(ap, dtype=np.float64).reshape(3)
            except Exception:  # pragma: no cover - defensive
                pass
        return np.zeros(3, dtype=np.float64)

    def capture(self) -> np.ndarray:
        """Generate a synthetic BGR frame with the laser line on the virtual piece.

        Returns:
            BGR image of shape (H, W, 3), dtype uint8.
        """
        self._ensure_model()
        height, width = self._height, self._width
        frame = np.random.randint(0, 8, (height, width, 3), dtype=np.uint8)

        if not self._geometric:
            return self._fallback_render(frame)

        assert self._surface is not None and self._plane is not None
        assert self._R is not None and self._t is not None and self._K is not None

        theta = float(self._rotation_angle_rad)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # P_platform = axis + Ry(-theta) @ P_object  (inverse of triangulate)
        ry_t = np.array(
            [[cos_t, 0.0, -sin_t], [0.0, 1.0, 0.0], [sin_t, 0.0, cos_t]],
            dtype=np.float64,
        )
        p_platform = self._axis + (ry_t @ self._surface.T).T

        # Keep only points lying within the thin laser sheet.
        normal = self._plane[:3]
        residual = p_platform @ normal + self._plane[3]
        in_sheet = np.abs(residual) < _LASER_SLICE_MM
        p_platform = p_platform[in_sheet]
        if p_platform.shape[0] == 0:
            return frame

        # Project into the camera frame: P_cam = R^T (P_platform - t).
        p_cam = (self._R.T @ (p_platform - self._t).T).T
        front = p_cam[:, 2] > 1e-3
        p_cam = p_cam[front]
        if p_cam.shape[0] == 0:
            return frame

        import cv2  # type: ignore[import]

        uv, _ = cv2.projectPoints(
            p_cam.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self._K,
            self._dist,
        )
        uv = uv.reshape(-1, 2)
        depth = p_cam[:, 2]

        cols = np.round(uv[:, 0]).astype(np.int64)
        rows = uv[:, 1]
        valid = (cols >= 0) & (cols < width) & (rows >= -4.0) & (rows < height + 4.0)
        cols, rows, depth = cols[valid], rows[valid], depth[valid]

        # Occlusion: keep the nearest surface point per image column (front face).
        nearest: dict[int, float] = {}
        nearest_depth: dict[int, float] = {}
        for u_px, v_px, d in zip(cols.tolist(), rows.tolist(), depth.tolist()):
            if u_px not in nearest_depth or d < nearest_depth[u_px]:
                nearest_depth[u_px] = d
                nearest[u_px] = v_px

        for u_px, v_px in nearest.items():
            row = int(round(v_px))
            for dr in range(-3, 4):
                r = row + dr
                if 0 <= r < height:
                    intensity = int(220 * math.exp(-0.5 * (dr / 1.5) ** 2))
                    frame[r, u_px, 1] = min(255, max(int(frame[r, u_px, 1]), intensity))

        logger.debug(
            "MockCamera.capture() id=%s → %d lit columns at angle=%.3f rad",
            self._camera_id,
            len(nearest),
            theta,
        )
        return frame

    def _fallback_render(self, frame: np.ndarray) -> np.ndarray:
        """Simple horizontal laser line when no calibration is available."""
        row = self._height // 2
        for u_px in range(self._width):
            for dr in range(-3, 4):
                r = row + dr
                if 0 <= r < self._height:
                    intensity = int(200 * math.exp(-0.5 * (dr / 1.5) ** 2))
                    frame[r, u_px, 1] = min(255, intensity)
        return frame


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
