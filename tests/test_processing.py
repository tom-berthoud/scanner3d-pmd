"""tests.test_processing - Unit tests for laser line extraction and triangulation."""

import math

import numpy as np
import pytest

from scanner.processing import extract_laser_line, triangulate
from tests.fixtures.generate import make_laser_frame_polyline, make_laser_frame_vertical


class TestExtractLaserLine:
    """Tests for scanner.processing.laser_line.extract_laser_line."""

    def test_detects_one_point_per_lit_row(self) -> None:
        frame = make_laser_frame_vertical(width=200, height=160, col=95.0, noise_amplitude=0)
        result = extract_laser_line(frame, threshold=100, min_pixels=10)
        assert result.shape[1] == 2
        assert result.shape[0] >= 145

    def test_column_mean_accuracy(self) -> None:
        true_col = 95.0
        frame = make_laser_frame_vertical(width=200, height=160, col=true_col, noise_amplitude=0)
        result = extract_laser_line(frame, threshold=80, min_pixels=10)
        assert result.shape[0] > 0
        assert abs(float(result[:, 0].mean()) - true_col) < 2.0

    def test_threshold_rejects_dark_pixels(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, 10, 1] = 90
        assert extract_laser_line(frame, threshold=100, min_pixels=1).shape == (0, 2)
        assert extract_laser_line(frame, threshold=80, min_pixels=1).shape[0] == 20

    def test_no_line_returns_empty(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = extract_laser_line(frame, threshold=100, min_pixels=10)
        assert result.shape == (0, 2)

    def test_below_min_pixels_returns_empty(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[120:125, 160, 1] = 220
        result = extract_laser_line(frame, threshold=100, min_pixels=20)
        assert result.shape == (0, 2)

    def test_wrong_shape_logs_warning(self) -> None:
        frame = np.zeros((240, 320), dtype=np.uint8)
        result = extract_laser_line(frame, threshold=100)
        assert result.shape == (0, 2)

    def test_col_values_in_range(self) -> None:
        frame = make_laser_frame_vertical(width=200, height=150, col=75.0)
        result = extract_laser_line(frame, threshold=80, min_pixels=5)
        assert result[:, 0].min() >= 0
        assert result[:, 0].max() < 200

    def test_row_values_in_range(self) -> None:
        frame = make_laser_frame_vertical(width=200, height=150, col=75.0)
        result = extract_laser_line(frame, threshold=80, min_pixels=5)
        assert result[:, 1].min() >= 0
        assert result[:, 1].max() < 150

    def test_legacy_mode_argument_still_uses_row_mean(self) -> None:
        true_col = 95.25
        frame = make_laser_frame_vertical(width=200, height=160, col=true_col, noise_amplitude=0)
        result = extract_laser_line(
            frame,
            threshold=80,
            min_pixels=20,
            subpixel=True,
            mode="row_green",
        )
        assert result.shape[0] >= 145
        assert abs(float(result[:, 0].mean()) - true_col) < 1.0
        assert float(result[:, 1].max() - result[:, 1].min()) > 140.0

    def test_detects_bent_polyline_by_row_mean(self) -> None:
        frame = make_laser_frame_polyline(
            points=[(108.0, 20.0), (122.0, 80.0), (94.0, 145.0)],
            width=220,
            height=180,
            noise_amplitude=0,
        )
        result = extract_laser_line(frame, threshold=80, min_pixels=20)
        assert result.shape[0] >= 100
        assert float(result[:, 0].max() - result[:, 0].min()) >= 25.0
        assert float(result[:, 1].max() - result[:, 1].min()) >= 110.0

    def test_mask_rectangles_remove_rows(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, 10, 1] = 220
        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            mask_rects=[[0, 5, 30, 15]],
        )
        assert result.shape[0] == 10
        assert set(result[:, 1].astype(int).tolist()) == set(range(5)) | set(range(15, 20))

    def test_polygon_mask_removes_trapezoid_area(self) -> None:
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        frame[:, 20, 1] = 220
        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            mask_rects=[[[15, 5], [25, 8], [23, 22], [17, 24]]],
        )
        rows = set(result[:, 1].astype(int).tolist())
        assert 12 not in rows
        assert 16 not in rows
        assert 2 in rows
        assert 27 in rows

    def test_masks_snap_near_image_edges(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, 0, 1] = 220
        frame[:, 29, 1] = 220

        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            mask_rects=[[1, 2, 29, 18]],
        )

        assert result.shape == (0, 2)

    def test_masks_do_not_snap_far_from_image_edges(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[455, 200, 1] = 220
        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            mask_rects=[[0, 451, 414, 480]],
        )
        assert result.shape == (0, 2)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[445, 200, 1] = 220
        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            mask_rects=[[0, 451, 414, 480]],
        )
        assert result.shape == (1, 2)

    def test_y_stride_reduces_detected_rows_in_original_coordinates(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, 120, 1] = 220

        result = extract_laser_line(frame, threshold=100, min_pixels=1, y_stride=2)

        assert result.shape[0] == 240
        assert result[:, 1].min() == 0.0
        assert result[:, 1].max() == 478.0
        assert np.all((result[:, 1] % 2) == 0)

    def test_x_stride_keeps_original_column_coordinates(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, 12, 1] = 220

        result = extract_laser_line(frame, threshold=100, min_pixels=1, x_stride=3)

        assert result.shape[0] == 20
        assert np.all(result[:, 0] == 12.0)

    def test_sampled_rectangle_mask_uses_original_coordinates(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        frame[:, 12, 1] = 220

        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            x_stride=3,
            y_stride=2,
            mask_rects=[[0, 4, 30, 12]],
        )

        assert result.shape[0] == 6
        assert set(result[:, 1].astype(int).tolist()) == {0, 2, 12, 14, 16, 18}

    def test_sampled_polygon_mask_uses_original_coordinates(self) -> None:
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        frame[:, 18, 1] = 220

        result = extract_laser_line(
            frame,
            threshold=100,
            min_pixels=1,
            x_stride=2,
            y_stride=3,
            mask_rects=[[[14, 6], [24, 6], [24, 21], [14, 21]]],
        )

        rows = set(result[:, 1].astype(int).tolist())
        assert 9 not in rows
        assert 18 not in rows
        assert 0 in rows
        assert 24 in rows


def _make_camera_matrix(
    fx: float = 800.0, fy: float = 800.0, cx: float = 320.0, cy: float = 240.0
) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _zero_dist() -> np.ndarray:
    return np.zeros(5, dtype=np.float64)


class TestTriangulate:
    """Tests for scanner.processing.triangulation.triangulate."""

    def test_empty_input_returns_empty(self) -> None:
        result = triangulate(
            np.empty((0, 2), dtype=np.float32),
            _make_camera_matrix(),
            _zero_dist(),
            np.array([0.0, 0.0, 1.0, -300.0]),
            0.0,
        )
        assert result.shape == (0, 3)

    def test_output_shape(self) -> None:
        line = np.column_stack(
            [np.arange(10, dtype=np.float32), np.full(10, 240.0, dtype=np.float32)]
        )
        result = triangulate(
            line,
            _make_camera_matrix(),
            _zero_dist(),
            np.array([0.0, 0.0, 1.0, -300.0]),
            0.0,
        )
        assert result.shape[1] == 3

    def test_known_geometry(self) -> None:
        line = np.array([[320.0, 240.0]], dtype=np.float32)
        result = triangulate(
            line,
            _make_camera_matrix(cx=320.0, cy=240.0),
            _zero_dist(),
            np.array([0.0, 0.0, 1.0, -300.0]),
            0.0,
        )
        assert result.shape == (1, 3)
        assert abs(result[0, 2] - 300.0) < 1e-6
        assert abs(result[0, 0]) < 1e-6
        assert abs(result[0, 1]) < 1e-6

    def test_rotation_identity(self) -> None:
        line = np.array([[320.0, 240.0]], dtype=np.float32)
        r0 = triangulate(
            line, _make_camera_matrix(), _zero_dist(), np.array([0, 0, 1, -300.0]), 0.0
        )
        r_pi = triangulate(
            line,
            _make_camera_matrix(),
            _zero_dist(),
            np.array([0, 0, 1, -300.0]),
            2 * math.pi,
        )
        np.testing.assert_allclose(r0, r_pi, atol=1e-6)

    def test_parallel_rays_filtered(self) -> None:
        line = np.array([[320.0, 240.0]], dtype=np.float32)
        result = triangulate(
            line,
            _make_camera_matrix(cx=320.0, cy=240.0),
            _zero_dist(),
            np.array([0.0, 1.0, 0.0, 0.0]),
            0.0,
        )
        assert result.shape[0] == 0

    def test_camera_to_platform_translation(self) -> None:
        line = np.array([[320.0, 240.0]], dtype=np.float32)
        result = triangulate(
            line,
            _make_camera_matrix(cx=320.0, cy=240.0),
            _zero_dist(),
            np.array([0.0, 0.0, 1.0, -300.0]),
            0.0,
            camera_to_platform_translation=np.array([10.0, 20.0, 30.0]),
        )
        np.testing.assert_allclose(result[0], np.array([10.0, 20.0, 300.0]), atol=1e-6)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            triangulate(
                np.zeros((5, 3)),
                _make_camera_matrix(),
                _zero_dist(),
                np.array([0, 0, 1, -300.0]),
                0.0,
            )
