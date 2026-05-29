"""scanner.hardware.door — Safety door interlock sensor.

SAFETY RULES (non-negotiable, agents.md section 8):
- When the interlock is enabled and the door opens, the scan must stop
  immediately and the laser must be turned off.
- The interlock can be enabled/disabled quickly from config
  (``safety.door_interlock.enabled``) without rewiring.

Typical wiring: a magnetic reed switch or microswitch between the GPIO pin
and GND, using the internal pull-up. Door closed → switch closed → pin LOW
(``is_pressed`` is True). Door open → switch open → pin HIGH. The
``closed_when_pressed`` flag inverts this for the opposite wiring.
"""

import logging

logger = logging.getLogger(__name__)


class DoorSensor:
    """GPIO input driver for the safety door interlock switch.

    When ``enabled`` is False the sensor never touches the GPIO and
    :meth:`is_open` always returns False, so the interlock is effectively
    bypassed. This is the "quick disable" toggle.

    Args:
        config: ``safety.door_interlock`` configuration dict with keys:
            - enabled: master on/off switch for the interlock (default False)
            - gpio_pin: BCM GPIO pin number for the door switch (default 23)
            - pull_up: enable the internal pull-up resistor (default True)
            - closed_when_pressed: True if a *pressed* switch (pin pulled to
              the active level) means the door is closed (default True)
            - bounce_time_s: debounce time for the switch (default 0.05)

    Raises:
        HardwareError: if the interlock is enabled but gpiozero is
            unavailable or the pin setup fails.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        config = config or {}
        self._enabled: bool = bool(config.get("enabled", False))
        self._gpio_pin: int = int(config.get("gpio_pin", 23))
        self._closed_when_pressed: bool = bool(config.get("closed_when_pressed", True))
        self._button = None

        if not self._enabled:
            logger.info("Door interlock disabled by config (no GPIO claimed)")
            return

        try:
            from gpiozero import Button  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("gpiozero not available for door interlock") from exc

        pull_up = bool(config.get("pull_up", True))
        bounce_time = config.get("bounce_time_s", 0.05)
        bounce_time = float(bounce_time) if bounce_time else None

        try:
            self._button = Button(self._gpio_pin, pull_up=pull_up, bounce_time=bounce_time)
            logger.info(
                "Door interlock initialised on GPIO %d (pull_up=%s, closed_when_pressed=%s)",
                self._gpio_pin,
                pull_up,
                self._closed_when_pressed,
            )
        except Exception as exc:
            raise HardwareError(f"Door sensor GPIO init failed: {exc}") from exc

    @property
    def enabled(self) -> bool:
        """True when the interlock is active (config + GPIO claimed)."""
        return self._enabled

    def is_open(self) -> bool:
        """Return True when the door is open *and* the interlock is enabled.

        When the interlock is disabled this always returns False so callers
        never block on it.
        """
        if not self._enabled or self._button is None:
            return False
        pressed = bool(self._button.is_pressed)
        return (not pressed) if self._closed_when_pressed else pressed

    def close(self) -> None:
        """Release the GPIO resources."""
        if self._button is not None:
            try:
                self._button.close()
                logger.info("Door interlock GPIO released")
            except Exception as exc:
                logger.warning("Error closing door sensor GPIO: %s", exc)
            finally:
                self._button = None
