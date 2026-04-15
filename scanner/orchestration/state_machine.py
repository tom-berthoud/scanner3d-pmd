"""scanner.orchestration.state_machine — Scanner finite-state machine.

Defines the ScannerState enum and the StateMachine class that enforces
valid transitions as documented in agents.md section 7.

Valid transitions:
    IDLE        → SCANNING, CALIBRATING
    CALIBRATING → IDLE, ERROR
    SCANNING    → PROCESSING, ERROR
    PROCESSING  → EXPORTING, COMPLETE, ERROR
    EXPORTING   → COMPLETE, ERROR
    COMPLETE    → IDLE, ERROR
    ERROR       → IDLE
    (any state) → ERROR
"""

import logging
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Valid transitions table
# --------------------------------------------------------------------------- #

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "IDLE": {"SCANNING", "CALIBRATING", "ERROR"},
    "CALIBRATING": {"IDLE", "ERROR"},
    "SCANNING": {"PROCESSING", "ERROR"},
    "PROCESSING": {"EXPORTING", "COMPLETE", "ERROR"},
    "EXPORTING": {"COMPLETE", "ERROR"},
    "COMPLETE": {"IDLE", "ERROR"},
    "ERROR": {"IDLE"},
}


class ScannerState(Enum):
    """Enumeration of all valid scanner states.

    Meanings and associated LED patterns are defined in agents.md §7.
    """

    IDLE = auto()
    CALIBRATING = auto()
    SCANNING = auto()
    PROCESSING = auto()
    EXPORTING = auto()
    COMPLETE = auto()
    ERROR = auto()


class StateMachine:
    """Finite-state machine for the 3D scanner.

    Enforces the transition table from agents.md §7 and notifies a list
    of observer callbacks on every successful transition.

    Args:
        initial_state: Starting state (default IDLE).

    Example::

        sm = StateMachine()
        sm.add_observer(lambda old, new: print(f"{old} → {new}"))
        sm.transition(ScannerState.SCANNING)
    """

    def __init__(self, initial_state: ScannerState = ScannerState.IDLE) -> None:
        self._state: ScannerState = initial_state
        self._observers: list[Callable[[ScannerState, ScannerState], None]] = []
        logger.info("StateMachine initialised in state %s", initial_state.name)

    @property
    def current_state(self) -> ScannerState:
        """The current state of the machine."""
        return self._state

    def add_observer(
        self, callback: Callable[[ScannerState, ScannerState], None]
    ) -> None:
        """Register an observer that is called on every state transition.

        The callback receives (old_state, new_state).

        Args:
            callback: Callable accepting two ScannerState arguments.
        """
        self._observers.append(callback)
        logger.debug("StateMachine: observer registered (%s)", callback)

    def remove_observer(
        self, callback: Callable[[ScannerState, ScannerState], None]
    ) -> None:
        """Remove a previously registered observer.

        Args:
            callback: The callable to remove.  Silently ignored if not found.
        """
        try:
            self._observers.remove(callback)
        except ValueError:
            pass

    def transition(self, new_state: ScannerState) -> None:
        """Move to *new_state* if the transition is valid.

        Args:
            new_state: Target state.

        Raises:
            ValueError: if the transition from the current state to
                *new_state* is not listed in the valid transitions table.
        """
        old_name = self._state.name
        new_name = new_state.name

        allowed = _VALID_TRANSITIONS.get(old_name, set())
        if new_name not in allowed:
            raise ValueError(
                f"Invalid transition: {old_name} → {new_name}. "
                f"Allowed from {old_name}: {sorted(allowed)}"
            )

        old_state = self._state
        self._state = new_state
        logger.info("State transition: %s → %s", old_name, new_name)

        for observer in list(self._observers):
            try:
                observer(old_state, new_state)
            except Exception as exc:
                logger.error(
                    "Observer %s raised an exception during transition %s→%s: %s",
                    observer,
                    old_name,
                    new_name,
                    exc,
                )

    def can_transition_to(self, new_state: ScannerState) -> bool:
        """Check if a transition to *new_state* is currently valid.

        Args:
            new_state: State to test.

        Returns:
            True if the transition is allowed, False otherwise.
        """
        allowed = _VALID_TRANSITIONS.get(self._state.name, set())
        return new_state.name in allowed

    def reset(self) -> None:
        """Force the state machine back to IDLE without triggering observers.

        Use only in emergency shutdown or test teardown scenarios.
        """
        logger.warning(
            "StateMachine.reset(): forcing state from %s to IDLE (no observers notified)",
            self._state.name,
        )
        self._state = ScannerState.IDLE
