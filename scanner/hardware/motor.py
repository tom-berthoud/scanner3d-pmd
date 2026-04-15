"""scanner.hardware.motor — NEMA 17 stepper motor driver via DM320T.

Controls the stepper motor using two GPIO output pins: STEP and DIR.
Supports progressive acceleration and deceleration to prevent mechanical
shock (see agents.md section 8 safety requirements).
"""

import logging
import time

logger = logging.getLogger(__name__)


class StepperMotor:
    """NEMA 17 stepper motor driver for DM320T STEP/DIR interface.

    The DM320T driver uses:
        - STEP pin: rising edge advances one micro-step
        - DIR  pin: HIGH = clockwise, LOW = counter-clockwise
        - ENABLE pin: active LOW (LOW = driver enabled)

    Progressive acceleration ramps are applied to avoid sudden torque
    changes that could cause missed steps or mechanical damage.

    Args:
        config: Motor configuration dict with keys:
            - step_pin: GPIO BCM pin number for STEP signal
            - dir_pin: GPIO BCM pin number for DIR signal
            - enable_pin: GPIO BCM pin number for ENABLE signal
            - steps_per_rev: full steps per motor revolution (default 200)
            - microstepping: microstepping factor (default 1)
            - speed_rpm: target speed in RPM (default 5)
            - accel_steps: number of steps over which to accelerate (default 20)

    Raises:
        HardwareError: if gpiozero is unavailable or pin setup fails.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        try:
            from gpiozero import OutputDevice  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("gpiozero not available") from exc

        self._step_pin: int = int(config.get("step_pin", 23))
        self._dir_pin: int = int(config.get("dir_pin", 24))
        self._enable_pin: int = int(config.get("enable_pin", 25))
        self._steps_per_rev: int = int(config.get("steps_per_rev", 200))
        self._microstepping: int = int(config.get("microstepping", 1))
        self._speed_rpm: float = float(config.get("speed_rpm", 5.0))
        self._accel_steps: int = int(config.get("accel_steps", 20))

        # Steps per revolution accounting for microstepping
        self._effective_steps: int = self._steps_per_rev * self._microstepping

        # Base step delay in seconds for the target speed
        self._base_delay_s: float = 60.0 / (self._speed_rpm * self._effective_steps)

        try:
            self._step = OutputDevice(self._step_pin, active_high=True, initial_value=False)
            self._direction = OutputDevice(self._dir_pin, active_high=True, initial_value=False)
            self._enable = OutputDevice(
                self._enable_pin, active_high=False, initial_value=True
            )  # active LOW → starts enabled
            self._initialised: bool = True
            logger.info(
                "StepperMotor initialised (step=%d, dir=%d, enable=%d, "
                "steps/rev=%d, µstep=%d, speed=%.1f RPM)",
                self._step_pin,
                self._dir_pin,
                self._enable_pin,
                self._steps_per_rev,
                self._microstepping,
                self._speed_rpm,
            )
        except Exception as exc:
            raise HardwareError(f"Motor GPIO init failed: {exc}") from exc

    def motor_step(self, n: int, direction: str = "clockwise") -> None:
        """Advance the motor *n* steps with progressive acceleration/deceleration.

        The step delay is linearly interpolated from 3× the base delay down
        to the base delay over *accel_steps* steps, then back up to 3× over
        the last *accel_steps* steps (deceleration).

        Args:
            n: Number of steps to advance (positive integer).
            direction: 'clockwise' or 'counterclockwise'.

        Raises:
            HardwareError: if the motor is not properly initialised.
            ValueError: if *direction* is not recognised.
        """
        from scanner.hardware import HardwareError

        if not self._initialised:
            raise HardwareError("Motor not initialised")
        if direction not in ("clockwise", "counterclockwise"):
            raise ValueError(f"Unknown direction: {direction!r}")

        # Set direction pin
        if direction == "clockwise":
            self._direction.on()
        else:
            self._direction.off()

        accel = min(self._accel_steps, n // 2)
        max_delay = self._base_delay_s * 3.0
        min_delay = self._base_delay_s

        for i in range(n):
            # Compute current delay with trapezoidal profile
            if i < accel:
                # Acceleration phase
                delay = max_delay - (max_delay - min_delay) * (i / max(accel, 1))
            elif i >= n - accel:
                # Deceleration phase
                steps_from_end = n - 1 - i
                delay = max_delay - (max_delay - min_delay) * (steps_from_end / max(accel, 1))
            else:
                delay = min_delay

            # Generate STEP pulse (minimum 1 µs high time for DM320T)
            self._step.on()
            time.sleep(max(delay / 2, 1e-6))
            self._step.off()
            time.sleep(max(delay / 2, 1e-6))

        logger.debug("StepperMotor.motor_step(n=%d, dir=%s) done", n, direction)

    def enable(self) -> None:
        """Enable the motor driver (hold torque)."""
        self._enable.on()
        logger.debug("StepperMotor enabled")

    def disable(self) -> None:
        """Disable the motor driver (free-wheel mode, reduces heat)."""
        self._enable.off()
        logger.debug("StepperMotor disabled (free-wheel)")

    def close(self) -> None:
        """Release GPIO resources."""
        try:
            self.disable()
            self._step.close()
            self._direction.close()
            self._enable.close()
            logger.info("StepperMotor GPIO released")
        except Exception as exc:
            logger.warning("Error closing motor GPIO: %s", exc)
