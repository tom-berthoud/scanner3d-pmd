"""scanner.hardware.led — RGB LED controller via GPIO.

Supports setting individual LEDs on/off and blinking them at a given
frequency using background threads.  LED meanings per state are defined
in agents.md section 7.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Default GPIO pin mapping for the three LEDs
_DEFAULT_PINS: dict[str, int] = {
    "green": 27,
    "orange": 22,
    "red": 5,
}


class LED:
    """Multi-colour LED controller using gpiozero OutputDevice.

    Args:
        config: Interface configuration dict.  Optionally contains
            led_pins: {color: gpio_pin} mapping to override defaults.

    Raises:
        HardwareError: if gpiozero is unavailable or a pin fails to init.
    """

    def __init__(self, config: dict) -> None:
        from scanner.hardware import HardwareError

        try:
            from gpiozero import OutputDevice  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("gpiozero not available") from exc

        pin_map: dict[str, int] = config.get("led_pins", _DEFAULT_PINS)
        self._outputs: dict[str, "OutputDevice"] = {}
        self._blink_threads: dict[str, threading.Thread] = {}
        self._blink_stop_events: dict[str, threading.Event] = {}

        try:
            for color, pin in pin_map.items():
                self._outputs[color] = OutputDevice(pin, active_high=True, initial_value=False)
            logger.info("LED controller initialised (pins: %s)", pin_map)
        except Exception as exc:
            raise HardwareError(f"LED GPIO init failed: {exc}") from exc

    def led_set(self, color: str, state: bool) -> None:
        """Set an LED on or off, stopping any active blink.

        Args:
            color: LED colour key ('green', 'orange', 'red').
            state: True = on, False = off.

        Raises:
            HardwareError: if *color* is unknown.
        """
        from scanner.hardware import HardwareError

        self._stop_blink(color)
        output = self._outputs.get(color)
        if output is None:
            raise HardwareError(f"Unknown LED colour: {color!r}")
        if state:
            output.on()
        else:
            output.off()
        logger.debug("LED %s → %s", color, "ON" if state else "OFF")

    def led_blink(self, color: str, frequency_hz: float) -> None:
        """Blink an LED at *frequency_hz* Hz.

        A frequency ≤ 0 stops blinking and turns the LED off.

        Args:
            color: LED colour key.
            frequency_hz: Blink frequency in Hz.

        Raises:
            HardwareError: if *color* is unknown.
        """
        from scanner.hardware import HardwareError

        self._stop_blink(color)
        output = self._outputs.get(color)
        if output is None:
            raise HardwareError(f"Unknown LED colour: {color!r}")

        if frequency_hz <= 0:
            output.off()
            return

        stop_event = threading.Event()
        self._blink_stop_events[color] = stop_event
        half_period = 0.5 / frequency_hz

        def _blink_loop() -> None:
            while not stop_event.is_set():
                output.on()
                stop_event.wait(timeout=half_period)
                if stop_event.is_set():
                    break
                output.off()
                stop_event.wait(timeout=half_period)
            output.off()

        thread = threading.Thread(target=_blink_loop, daemon=True, name=f"led-blink-{color}")
        self._blink_threads[color] = thread
        thread.start()
        logger.debug("LED %s blinking at %.2f Hz", color, frequency_hz)

    def _stop_blink(self, color: str) -> None:
        """Stop any active blink thread for *color*.

        Args:
            color: LED colour key.
        """
        event = self._blink_stop_events.pop(color, None)
        if event is not None:
            event.set()
        thread = self._blink_threads.pop(color, None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def all_off(self) -> None:
        """Turn off all LEDs and stop all blink threads."""
        for color in list(self._outputs.keys()):
            self._stop_blink(color)
            self._outputs[color].off()
        logger.debug("All LEDs off")

    def close(self) -> None:
        """Release all LED GPIO resources."""
        self.all_off()
        for output in self._outputs.values():
            try:
                output.close()
            except Exception as exc:
                logger.warning("Error closing LED output: %s", exc)
        logger.info("LED controller GPIO released")
