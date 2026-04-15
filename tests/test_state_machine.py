"""tests.test_state_machine — Unit tests for the scanner state machine."""

import pytest

from scanner.orchestration.state_machine import ScannerState, StateMachine


class TestStateMachineInit:
    """Initialisation tests."""

    def test_default_state_is_idle(self) -> None:
        sm = StateMachine()
        assert sm.current_state == ScannerState.IDLE

    def test_custom_initial_state(self) -> None:
        sm = StateMachine(initial_state=ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR


class TestValidTransitions:
    """All valid transitions from agents.md §7."""

    def test_idle_to_scanning(self) -> None:
        sm = StateMachine()
        sm.transition(ScannerState.SCANNING)
        assert sm.current_state == ScannerState.SCANNING

    def test_idle_to_calibrating(self) -> None:
        sm = StateMachine()
        sm.transition(ScannerState.CALIBRATING)
        assert sm.current_state == ScannerState.CALIBRATING

    def test_idle_to_error(self) -> None:
        sm = StateMachine()
        sm.transition(ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR

    def test_calibrating_to_idle(self) -> None:
        sm = StateMachine(ScannerState.CALIBRATING)
        sm.transition(ScannerState.IDLE)
        assert sm.current_state == ScannerState.IDLE

    def test_calibrating_to_error(self) -> None:
        sm = StateMachine(ScannerState.CALIBRATING)
        sm.transition(ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR

    def test_scanning_to_processing(self) -> None:
        sm = StateMachine(ScannerState.SCANNING)
        sm.transition(ScannerState.PROCESSING)
        assert sm.current_state == ScannerState.PROCESSING

    def test_scanning_to_error(self) -> None:
        sm = StateMachine(ScannerState.SCANNING)
        sm.transition(ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR

    def test_processing_to_exporting(self) -> None:
        sm = StateMachine(ScannerState.PROCESSING)
        sm.transition(ScannerState.EXPORTING)
        assert sm.current_state == ScannerState.EXPORTING

    def test_processing_to_complete(self) -> None:
        sm = StateMachine(ScannerState.PROCESSING)
        sm.transition(ScannerState.COMPLETE)
        assert sm.current_state == ScannerState.COMPLETE

    def test_processing_to_error(self) -> None:
        sm = StateMachine(ScannerState.PROCESSING)
        sm.transition(ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR

    def test_exporting_to_complete(self) -> None:
        sm = StateMachine(ScannerState.EXPORTING)
        sm.transition(ScannerState.COMPLETE)
        assert sm.current_state == ScannerState.COMPLETE

    def test_exporting_to_error(self) -> None:
        sm = StateMachine(ScannerState.EXPORTING)
        sm.transition(ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR

    def test_complete_to_idle(self) -> None:
        sm = StateMachine(ScannerState.COMPLETE)
        sm.transition(ScannerState.IDLE)
        assert sm.current_state == ScannerState.IDLE

    def test_complete_to_error(self) -> None:
        sm = StateMachine(ScannerState.COMPLETE)
        sm.transition(ScannerState.ERROR)
        assert sm.current_state == ScannerState.ERROR

    def test_error_to_idle(self) -> None:
        sm = StateMachine(ScannerState.ERROR)
        sm.transition(ScannerState.IDLE)
        assert sm.current_state == ScannerState.IDLE

    def test_full_scan_cycle(self) -> None:
        """Walk through the happy-path scan cycle."""
        sm = StateMachine()
        for state in [
            ScannerState.SCANNING,
            ScannerState.PROCESSING,
            ScannerState.EXPORTING,
            ScannerState.COMPLETE,
            ScannerState.IDLE,
        ]:
            sm.transition(state)
        assert sm.current_state == ScannerState.IDLE


class TestInvalidTransitions:
    """Invalid transitions must raise ValueError."""

    @pytest.mark.parametrize(
        "from_state, to_state",
        [
            (ScannerState.IDLE, ScannerState.PROCESSING),
            (ScannerState.IDLE, ScannerState.EXPORTING),
            (ScannerState.IDLE, ScannerState.COMPLETE),
            (ScannerState.SCANNING, ScannerState.IDLE),
            (ScannerState.SCANNING, ScannerState.CALIBRATING),
            (ScannerState.SCANNING, ScannerState.COMPLETE),
            (ScannerState.SCANNING, ScannerState.EXPORTING),
            (ScannerState.PROCESSING, ScannerState.IDLE),
            (ScannerState.PROCESSING, ScannerState.SCANNING),
            (ScannerState.PROCESSING, ScannerState.CALIBRATING),
            (ScannerState.EXPORTING, ScannerState.IDLE),
            (ScannerState.EXPORTING, ScannerState.SCANNING),
            (ScannerState.EXPORTING, ScannerState.PROCESSING),
            (ScannerState.COMPLETE, ScannerState.SCANNING),
            (ScannerState.COMPLETE, ScannerState.PROCESSING),
            (ScannerState.COMPLETE, ScannerState.EXPORTING),
            (ScannerState.ERROR, ScannerState.SCANNING),
            (ScannerState.ERROR, ScannerState.PROCESSING),
            (ScannerState.ERROR, ScannerState.COMPLETE),
        ],
    )
    def test_invalid_raises(
        self, from_state: ScannerState, to_state: ScannerState
    ) -> None:
        sm = StateMachine(initial_state=from_state)
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition(to_state)

    def test_state_unchanged_on_invalid(self) -> None:
        """State must not change if the transition is invalid."""
        sm = StateMachine()
        try:
            sm.transition(ScannerState.COMPLETE)
        except ValueError:
            pass
        assert sm.current_state == ScannerState.IDLE


class TestObservers:
    """Observer notification tests."""

    def test_observer_called_on_transition(self) -> None:
        calls: list[tuple] = []
        sm = StateMachine()
        sm.add_observer(lambda old, new: calls.append((old, new)))
        sm.transition(ScannerState.SCANNING)
        assert len(calls) == 1
        assert calls[0] == (ScannerState.IDLE, ScannerState.SCANNING)

    def test_observer_not_called_on_invalid(self) -> None:
        calls: list[tuple] = []
        sm = StateMachine()
        sm.add_observer(lambda old, new: calls.append((old, new)))
        try:
            sm.transition(ScannerState.COMPLETE)
        except ValueError:
            pass
        assert calls == []

    def test_multiple_observers(self) -> None:
        results_a: list[ScannerState] = []
        results_b: list[ScannerState] = []
        sm = StateMachine()
        sm.add_observer(lambda old, new: results_a.append(new))
        sm.add_observer(lambda old, new: results_b.append(new))
        sm.transition(ScannerState.SCANNING)
        assert results_a == [ScannerState.SCANNING]
        assert results_b == [ScannerState.SCANNING]

    def test_remove_observer(self) -> None:
        calls: list[tuple] = []

        def _cb(old: ScannerState, new: ScannerState) -> None:
            calls.append((old, new))

        sm = StateMachine()
        sm.add_observer(_cb)
        sm.remove_observer(_cb)
        sm.transition(ScannerState.SCANNING)
        assert calls == []

    def test_observer_exception_does_not_abort(self) -> None:
        """A crashing observer must not prevent the state transition."""

        def _bad_observer(old: ScannerState, new: ScannerState) -> None:
            raise RuntimeError("observer crash")

        sm = StateMachine()
        sm.add_observer(_bad_observer)
        sm.transition(ScannerState.SCANNING)  # must not raise
        assert sm.current_state == ScannerState.SCANNING


class TestCanTransitionTo:
    """Tests for can_transition_to helper."""

    def test_valid_returns_true(self) -> None:
        sm = StateMachine()
        assert sm.can_transition_to(ScannerState.SCANNING) is True

    def test_invalid_returns_false(self) -> None:
        sm = StateMachine()
        assert sm.can_transition_to(ScannerState.COMPLETE) is False


class TestReset:
    """Tests for emergency reset."""

    def test_reset_goes_to_idle(self) -> None:
        sm = StateMachine(ScannerState.ERROR)
        sm.reset()
        assert sm.current_state == ScannerState.IDLE

    def test_reset_does_not_call_observers(self) -> None:
        calls: list[tuple] = []
        sm = StateMachine(ScannerState.ERROR)
        sm.add_observer(lambda old, new: calls.append((old, new)))
        sm.reset()
        assert calls == []
