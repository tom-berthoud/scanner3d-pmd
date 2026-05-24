"""Tests for camera-scoped background filtering."""

from scanner.calibration.background_filter import background_crop_left_col


def test_background_crop_applies_to_matching_camera() -> None:
    cfg = {"enabled": True, "camera_id": "left", "crop_left_of_col": 120.0}

    assert background_crop_left_col(cfg, "left") == 120.0
    assert background_crop_left_col(cfg, "right") is None


def test_legacy_background_crop_does_not_apply_to_nappe_camera() -> None:
    cfg = {"enabled": True, "camera_id": None, "crop_left_of_col": 120.0}

    assert background_crop_left_col(cfg, "right") is None
    assert background_crop_left_col(cfg, "left") == 120.0
