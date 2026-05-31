"""scanner.interface.display_local — Local TFT display updater.

Bridges the state machine with the physical display.
"""

import logging

logger = logging.getLogger(__name__)


def update_display(state: str, progress: int) -> None:
    """Update the physical display to reflect *state* and *progress*.

    Args:
        state: Scanner state string (e.g. 'IDLE', 'SCANNING', 'ERROR').
        progress: Integer progress percentage 0–100.

    Note:
        This function silently ignores hardware errors to avoid masking the
        primary scan error.  Failures are logged at WARNING level.
    """
    _update_screen(state, progress)


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
