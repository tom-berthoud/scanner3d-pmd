"""scanner.hardware.laser — GPIO control for the VLM-520-56 laser line module.

SAFETY RULES (non-negotiable, agents.md section 8):
- The laser is disabled by default at startup.
- The laser must be disabled when transitioning to ERROR state.
- No calibration routine activates the laser without prior confirmation.
"""

import logging
import time

logger = logging.getLogger(__name__)


class Laser:
    """GPIO driver for the VLM-520-56 LPO-D45-F40 laser line module.

    The GPIO pin is configured as an active-HIGH output.  The laser is kept
    OFF at startup regardless of any previous state.

    Args:
        config: Laser configuration dict with keys:
            - gpio_pin: BCM GPIO pin number (default 17)
            - warmup_ms: stabilisation delay after switching on (default 50)

    Raises:
        HardwareError: if gpiozero is unavailable or pin setup fails.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        try:
            from gpiozero import OutputDevice  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("gpiozero not available") from exc

        self._gpio_pin: int = int(config.get("gpio_pin", 17))
        self._warmup_ms: int = int(config.get("warmup_ms", 50))

        try:
            # initial_value=False → laser OFF at startup (safety rule)
            self._output = OutputDevice(
                self._gpio_pin, active_high=True, initial_value=False
            )
            self._state: bool = False
            logger.info(
                "Laser initialised on GPIO %d (state=OFF, warmup=%d ms)",
                self._gpio_pin,
                self._warmup_ms,
            )
        except Exception as exc:
            raise HardwareError(f"Laser GPIO init failed: {exc}") from exc

    @property
    def state(self) -> bool:
        """Current laser state (True = on)."""
        return self._state

    def laser_set(self, state: bool) -> None:
        """Enable or disable the laser.

        When turning on, waits *warmup_ms* milliseconds for the laser diode
        to stabilise before returning.

        Args:
            state: True to activate the laser, False to deactivate.

        Raises:
            HardwareError: if the GPIO operation fails.
        """
        from scanner.hardware import HardwareError

        try:
            if state and not self._state:
                self._output.on()
                time.sleep(self._warmup_ms / 1000.0)
                logger.info("Laser ON (warmup %d ms elapsed)", self._warmup_ms)
            elif not state and self._state:
                self._output.off()
                logger.info("Laser OFF")
            self._state = state
        except Exception as exc:
            # Attempt emergency shutdown before re-raising
            try:
                self._output.off()
            except Exception:
                pass
            self._state = False
            raise HardwareError(f"Laser GPIO operation failed: {exc}") from exc

    def close(self) -> None:
        """Turn off the laser and release GPIO resources."""
        try:
            self._output.off()
            self._state = False
            self._output.close()
            logger.info("Laser GPIO released")
        except Exception as exc:
            logger.warning("Error closing laser GPIO: %s", exc)

    def __del__(self) -> None:
        """Emergency laser shutdown on garbage collection."""
        try:
            self._output.off()
        except Exception:
            pass
