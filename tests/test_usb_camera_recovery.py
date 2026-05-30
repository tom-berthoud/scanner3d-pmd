"""Tests for flaky USB camera recovery paths."""

import sys
from types import SimpleNamespace

import numpy as np


def test_camera_capture_all_retries_failed_configured_camera(monkeypatch) -> None:
    import scanner.hardware as hardware
    import scanner.hardware.usb_camera as usb_camera

    class FakeUSBCamera:
        def __init__(self, config: dict) -> None:
            self.config = config

        def capture(self) -> np.ndarray:
            return np.zeros((2, 3, 3), dtype=np.uint8)

    monkeypatch.setattr(hardware, "_ON_PI", True)
    monkeypatch.setattr(hardware, "_camera_instance", None)
    monkeypatch.setattr(hardware, "_camera_instances", {})
    monkeypatch.setattr(hardware, "_camera_configs", {"left": {"id": "left", "type": "usb"}})
    monkeypatch.setattr(
        hardware,
        "_failed_camera_configs",
        {"left": {"id": "left", "type": "usb"}},
    )
    monkeypatch.setattr(usb_camera, "USBCamera", FakeUSBCamera)

    frames = hardware.camera_capture_all()

    assert set(frames) == {"left"}
    assert frames["left"].shape == (2, 3, 3)


def test_usb_camera_reopens_after_failed_read(monkeypatch) -> None:
    from scanner.hardware.usb_camera import USBCamera

    frame = np.ones((2, 3, 3), dtype=np.uint8)
    captures = []

    class FakeCapture:
        def __init__(self, read_ok: bool) -> None:
            self.read_ok = read_ok
            self.released = False

        def isOpened(self) -> bool:
            return not self.released

        def set(self, _prop: int, _value: object) -> bool:
            return True

        def get(self, _prop: int) -> float:
            return 0.0

        def grab(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            return (True, frame.copy()) if self.read_ok else (False, None)

        def release(self) -> None:
            self.released = True

    def video_capture(_device: object, _backend: object = None) -> FakeCapture:
        cap = FakeCapture(read_ok=len(captures) > 0)
        captures.append(cap)
        return cap

    fake_cv2 = SimpleNamespace(
        CAP_PROP_AUTO_EXPOSURE=1,
        CAP_PROP_EXPOSURE=2,
        CAP_PROP_GAIN=3,
        CAP_PROP_BUFFERSIZE=4,
        CAP_PROP_FOURCC=5,
        CAP_PROP_FRAME_WIDTH=6,
        CAP_PROP_FRAME_HEIGHT=7,
        VideoCapture=video_capture,
        VideoWriter_fourcc=lambda *_chars: 0,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    cam = USBCamera(
        {
            "device_index": 0,
            "resolution": [3, 2],
            "startup_frame_check": False,
            "open_retries": 0,
            "reconnect_retries": 1,
            "reconnect_retry_delay_s": 0,
        }
    )

    captured = cam.capture()

    assert np.array_equal(captured, frame)
    assert len(captures) == 2
    assert captures[0].released
