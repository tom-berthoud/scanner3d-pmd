"""scanner.interface.display_local — Local TFT/LED display updater.

Bridges the state machine with the physical display and LEDs.
"""

import logging

logger = logging.getLogger(__name__)

# LED patterns per state: (color, mode)  mode = 'on' | 'blink_slow' | 'blink_fast' | 'off'
_STATE_LED_PATTERNS: dict[str, dict[str, tuple[str, str]]] = {
    "IDLE": {"orange": ("orange", "off"), "red": ("red", "off")},
    "CALIBRATING": {
        "orange": ("orange", "blink_slow"),
        "red": ("red", "off"),
    },
    "SCANNING": {
        "orange": ("orange", "on"),
        "red": ("red", "off"),
    },
    "PROCESSING": {
        "orange": ("orange", "blink_fast"),
        "red": ("red", "off"),
    },
    "EXPORTING": {
        "orange": ("orange", "blink_slow"),
        "red": ("red", "off"),
    },
    "COMPLETE": {"orange": ("orange", "off"), "red": ("red", "off")},
    "ERROR": {"orange": ("orange", "off"), "red": ("red", "on")},
}

_BLINK_FREQUENCIES: dict[str, float] = {
    "blink_slow": 1.0,  # Hz
    "blink_fast": 4.0,  # Hz
}


def update_display(state: str, progress: int) -> None:
    """Update the physical display and LEDs to reflect *state* and *progress*.

    Args:
        state: Scanner state string (e.g. 'IDLE', 'SCANNING', 'ERROR').
        progress: Integer progress percentage 0–100.

    Note:
        This function silently ignores hardware errors to avoid masking the
        primary scan error.  Failures are logged at WARNING level.
    """
    _update_leds(state)
    _update_screen(state, progress)


def _update_leds(state: str) -> None:
    """Apply LED pattern for *state*.

    Args:
        state: Scanner state string.
    """
    from scanner.hardware import HardwareError, led_set, led_blink

    pattern = _STATE_LED_PATTERNS.get(state, _STATE_LED_PATTERNS["IDLE"])
    for color, (_, mode) in pattern.items():
        try:
            if mode == "on":
                led_set(color, True)
            elif mode == "off":
                led_set(color, False)
            elif mode in _BLINK_FREQUENCIES:
                led_blink(color, _BLINK_FREQUENCIES[mode])
        except HardwareError as exc:
            logger.warning("LED update failed for %s/%s: %s", color, mode, exc)


def _update_screen(state: str, progress: int) -> None:
    """Write state and progress to the physical display.

    Args:
        state: Scanner state string.
        progress: 0–100 integer.
    """
    from scanner.hardware import HardwareError, display_text

    try:
        display_text(f"State: {state}", line=0)
        display_text(f"Progress: {progress:3d}%", line=1)
    except HardwareError as exc:
        logger.warning("Display update failed: %s", exc)
