"""tests.test_door_interlock — Safety door interlock tests.

These exercise the mock hardware path (no GPIO), validating the
enable/disable toggle, the open/closed reporting, and that an open door
aborts an acquisition sequence with the laser left OFF.
"""

import pytest
import yaml

import scanner.hardware as hw
from scanner.acquisition import run_capture_sequence_multi


def _load_config(enabled: bool) -> dict:
    with open("config/settings.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("safety", {}).setdefault("door_interlock", {})["enabled"] = enabled
    cfg["scan"]["n_steps"] = 6
    return cfg


class TestDoorSensorToggle:
    """The interlock can be turned on/off from config."""

    def test_disabled_never_reports_open(self) -> None:
        hw.init_hardware(_load_config(enabled=False))
        assert hw.door_interlock_enabled() is False
        # Even if the mock door is forced open, a disabled interlock stays quiet.
        hw._door_instance.set_open(True)
        assert hw.door_is_open() is False
        hw.check_door_interlock()  # must not raise

    def test_enabled_reports_state(self) -> None:
        hw.init_hardware(_load_config(enabled=True))
        assert hw.door_interlock_enabled() is True
        assert hw.door_is_open() is False
        hw._door_instance.set_open(True)
        assert hw.door_is_open() is True


class TestCheckInterlock:
    """check_door_interlock raises only when enabled and open."""

    def test_raises_when_open(self) -> None:
        hw.init_hardware(_load_config(enabled=True))
        hw._door_instance.set_open(True)
        with pytest.raises(hw.DoorOpenError):
            hw.check_door_interlock()

    def test_door_open_error_is_hardware_error(self) -> None:
        assert issubclass(hw.DoorOpenError, hw.HardwareError)


class TestAcquisitionAborts:
    """An open door aborts the capture sequence with the laser OFF."""

    def test_open_door_aborts_capture(self) -> None:
        cfg = _load_config(enabled=True)
        hw.init_hardware(cfg)
        hw._door_instance.set_open(True)
        with pytest.raises(hw.DoorOpenError):
            run_capture_sequence_multi(cfg["scan"]["n_steps"], cfg, save_frames=False)
        assert hw._laser_instance.state is False

    def test_closed_door_allows_capture(self) -> None:
        cfg = _load_config(enabled=True)
        hw.init_hardware(cfg)
        hw._door_instance.set_open(False)
        frames = run_capture_sequence_multi(cfg["scan"]["n_steps"], cfg, save_frames=False)
        assert all(len(v) == cfg["scan"]["n_steps"] for v in frames.values())
        assert hw._laser_instance.state is False
