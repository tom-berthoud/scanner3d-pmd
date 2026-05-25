"""tests.test_calibration — Unit tests for calibration file load/save."""

import os
import tempfile

import numpy as np
import pytest
import yaml

from scanner.calibration import CalibrationError, load_camera_calibration, load_laser_plane
from scanner.calibration.camera import _save_camera_calibration
from scanner.calibration.camera import approximate_camera_intrinsics
from scanner.calibration.camera import checkerboard_capture_quality
from scanner.calibration.laser_plane import _save_laser_plane
from scanner.calibration.laser_plane import calibrate_laser_plane_global_platform_z
from scanner.calibration.laser_plane import calibrate_laser_plane_platform_z


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_valid_intrinsics(path: str) -> None:
    data = {
        "camera_matrix": {
            "fx": 800.0,
            "fy": 800.0,
            "cx": 320.0,
            "cy": 240.0,
        },
        "dist_coeffs": [-0.35, 0.12, 0.0012, -0.0008, -0.04],
        "image_size": [640, 480],
        "rms_error": 0.32,
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)


def _write_valid_laser_plane(path: str) -> None:
    data = {
        "plane": {"a": 0.5, "b": 0.0, "c": 0.866, "d": -200.0},
        "triangulation_angle_deg": 30.0,
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)


# --------------------------------------------------------------------------- #
# load_camera_calibration
# --------------------------------------------------------------------------- #


class TestLoadCameraCalibration:
    """Tests for scanner.calibration.camera.load_camera_calibration."""

    def test_load_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cam.yaml")
            _write_valid_intrinsics(path)
            K, dc = load_camera_calibration(path)
        assert K.shape == (3, 3)
        assert dc.shape == (5,)

    def test_camera_matrix_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cam.yaml")
            _write_valid_intrinsics(path)
            K, dc = load_camera_calibration(path)
        assert abs(K[0, 0] - 800.0) < 1e-6, f"fx wrong: {K[0,0]}"
        assert abs(K[1, 1] - 800.0) < 1e-6, f"fy wrong: {K[1,1]}"
        assert abs(K[0, 2] - 320.0) < 1e-6, f"cx wrong: {K[0,2]}"
        assert abs(K[1, 2] - 240.0) < 1e-6, f"cy wrong: {K[1,2]}"
        assert K[2, 2] == 1.0

    def test_dist_coeffs_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cam.yaml")
            _write_valid_intrinsics(path)
            _, dc = load_camera_calibration(path)
        np.testing.assert_allclose(dc, [-0.35, 0.12, 0.0012, -0.0008, -0.04], atol=1e-9)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(CalibrationError, match="not found"):
            load_camera_calibration("/tmp/nonexistent_cam_file_xyz.yaml")

    def test_malformed_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.yaml")
            with open(path, "w") as fh:
                fh.write("camera_matrix:\n  not_fx: 1.0\n")
            with pytest.raises(CalibrationError):
                load_camera_calibration(path)

    def test_invalid_yaml_syntax_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.yaml")
            with open(path, "w") as fh:
                fh.write(": invalid: yaml: [\n")
            with pytest.raises(CalibrationError):
                load_camera_calibration(path)

    def test_dtype_is_float64(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cam.yaml")
            _write_valid_intrinsics(path)
            K, dc = load_camera_calibration(path)
        assert K.dtype == np.float64
        assert dc.dtype == np.float64


# --------------------------------------------------------------------------- #
# _save_camera_calibration (round-trip)
# --------------------------------------------------------------------------- #


class TestSaveLoadCameraCalibration:
    """Round-trip save/load tests."""

    def test_round_trip(self) -> None:
        K_orig = np.array([[850.0, 0, 310.0], [0, 845.0, 235.0], [0, 0, 1]], dtype=np.float64)
        dc_orig = np.array([-0.30, 0.10, 0.001, 0.002, -0.05], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.yaml")
            _save_camera_calibration(K_orig, dc_orig, (640, 480), 0.28, path)
            K, dc = load_camera_calibration(path)
        np.testing.assert_allclose(K, K_orig, atol=1e-6)
        np.testing.assert_allclose(dc, dc_orig, atol=1e-9)


# --------------------------------------------------------------------------- #
# load_laser_plane
# --------------------------------------------------------------------------- #


class TestLoadLaserPlane:
    """Tests for scanner.calibration.laser_plane.load_laser_plane."""

    def test_load_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lp.yaml")
            _write_valid_laser_plane(path)
            plane = load_laser_plane(path)
        assert plane.shape == (4,)

    def test_plane_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lp.yaml")
            _write_valid_laser_plane(path)
            plane = load_laser_plane(path)
        np.testing.assert_allclose(plane, [0.5, 0.0, 0.866, -200.0], atol=1e-6)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(CalibrationError, match="not found"):
            load_laser_plane("/tmp/nonexistent_lp_xyz.yaml")

    def test_malformed_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.yaml")
            with open(path, "w") as fh:
                fh.write("plane:\n  missing_a: 1.0\n")
            with pytest.raises(CalibrationError):
                load_laser_plane(path)

    def test_dtype_float64(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lp.yaml")
            _write_valid_laser_plane(path)
            plane = load_laser_plane(path)
        assert plane.dtype == np.float64

    def test_round_trip(self) -> None:
        plane_orig = np.array([0.48, 0.02, 0.877, -205.0], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rt.yaml")
            _save_laser_plane(plane_orig, 29.5, path)
            plane = load_laser_plane(path)
        np.testing.assert_allclose(plane, plane_orig, atol=1e-6)


class TestLaserPlanePlatformZCalibration:
    """Tests for vertical-board laser calibration in the platform frame."""

    def test_fits_plane_from_platform_z_boards(self) -> None:
        images = []
        for col in (40, 55, 70):
            img = np.zeros((80, 120, 3), dtype=np.uint8)
            img[:, col, 1] = 255
            images.append(img)

        camera_matrix = np.array(
            [[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            plane = calibrate_laser_plane_platform_z(
                images,
                [100.0, 120.0, 140.0],
                camera_matrix,
                np.zeros(5),
                np.eye(3),
                np.zeros(3),
                output_path=os.path.join(tmpdir, "laser.yaml"),
                threshold=100,
                min_pixels=5,
                mask_rects=[],
            )

        assert plane.shape == (4,)
        assert np.isfinite(plane).all()

    def test_fits_global_plane_from_two_camera_platform_z_boards(self) -> None:
        camera_matrix = np.array(
            [[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        observations = []
        for camera_id, translation in (
            ("right", np.array([20.0, 0.0, 0.0])),
            ("left", np.array([-20.0, 0.0, 0.0])),
        ):
            images = []
            for col in (55, 60, 65):
                img = np.zeros((80, 120, 3), dtype=np.uint8)
                img[:, col, 1] = 255
                images.append(img)
            observations.append(
                {
                    "camera_id": camera_id,
                    "reference_images": images,
                    "platform_z_mm": [100.0, 120.0, 140.0],
                    "camera_matrix": camera_matrix,
                    "dist_coeffs": np.zeros(5),
                    "camera_to_platform_rotation": np.eye(3),
                    "camera_to_platform_translation": translation,
                    "threshold": 100,
                    "min_pixels": 5,
                    "mask_rects": [],
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            plane = calibrate_laser_plane_global_platform_z(
                observations,
                output_path=os.path.join(tmpdir, "laser.yaml"),
            )

        assert plane.shape == (4,)
        assert np.isfinite(plane).all()


class TestApproximateCameraIntrinsics:
    """Tests for runtime intrinsics when checkerboard calibration is disabled."""

    def test_default_values(self) -> None:
        K, dc = approximate_camera_intrinsics((640, 480))
        assert K.shape == (3, 3)
        assert dc.shape == (5,)
        assert abs(K[0, 0] - 800.0) < 1e-6
        assert abs(K[1, 1] - 800.0) < 1e-6
        assert abs(K[0, 2] - 320.0) < 1e-6
        assert abs(K[1, 2] - 240.0) < 1e-6
        np.testing.assert_allclose(dc, np.zeros(5), atol=1e-12)

    def test_custom_focal_scale(self) -> None:
        K, _ = approximate_camera_intrinsics((1000, 500), focal_scale=1.1)
        assert abs(K[0, 0] - 1100.0) < 1e-6
        assert abs(K[1, 1] - 1100.0) < 1e-6
        assert abs(K[0, 2] - 500.0) < 1e-6
        assert abs(K[1, 2] - 250.0) < 1e-6

    def test_invalid_resolution_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid resolution"):
            approximate_camera_intrinsics((0, 480))

    def test_invalid_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="focal_scale"):
            approximate_camera_intrinsics((640, 480), focal_scale=0.0)


class TestCheckerboardCaptureQuality:
    """Tests for guided checkerboard capture quality checks."""

    def test_rejects_blank_image_without_checkerboard(self) -> None:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        quality = checkerboard_capture_quality(img, (9, 6))
        assert quality["found"] is False
        assert quality["accepted"] is False
        assert "damier introuvable" in quality["issues"]

    def test_rejects_overexposed_image(self) -> None:
        img = np.full((240, 320, 3), 255, dtype=np.uint8)
        quality = checkerboard_capture_quality(img, (9, 6))
        assert quality["accepted"] is False
        assert "surexpose" in quality["issues"]
