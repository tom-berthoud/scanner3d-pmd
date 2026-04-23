"""tests.test_processing — Unit tests for laser line extraction and triangulation."""

import math

import numpy as np
import pytest

from scanner.processing import extract_laser_line, triangulate
from tests.fixtures.generate import (
    make_laser_frame,
    make_laser_frame_polyline,
    make_laser_frame_vertical,
)


# --------------------------------------------------------------------------- #
# extract_laser_line
# --------------------------------------------------------------------------- #


class TestExtractLaserLine:
    """Tests for scanner.processing.laser_line.extract_laser_line."""

    def test_detects_horizontal_line(self) -> None:
        """A synthetic horizontal laser line should be detected."""
        frame = make_laser_frame(width=320, height=240, row=120.0)
        result = extract_laser_line(frame, threshold=100, min_pixels=10)
        assert result.shape[1] == 2, "Result must have 2 columns (col, row)"
        assert result.shape[0] >= 300, f"Expected ≥300 pixels detected, got {result.shape[0]}"

    def test_row_accuracy(self) -> None:
        """Detected row should be close to the ground-truth laser row."""
        true_row = 150.0
        frame = make_laser_frame(width=320, height=240, row=true_row, noise_amplitude=0)
        result = extract_laser_line(frame, threshold=80, min_pixels=10, subpixel=True)
        assert result.shape[0] > 0, "No line detected"
        mean_row = float(result[:, 1].mean())
        assert abs(mean_row - true_row) < 2.0, (
            f"Mean detected row {mean_row:.2f} differs from truth {true_row} by > 2 px"
        )

    def test_subpixel_vs_integer(self) -> None:
        """Sub-pixel detection should be at least as accurate as integer."""
        true_row = 100.5  # fractional row
        frame = make_laser_frame(width=320, height=240, row=true_row, noise_amplitude=0)
        result_sp = extract_laser_line(frame, threshold=80, min_pixels=10, subpixel=True)
        result_int = extract_laser_line(frame, threshold=80, min_pixels=10, subpixel=False)
        assert result_sp.shape[0] > 0
        assert result_int.shape[0] > 0
        err_sp = abs(float(result_sp[:, 1].mean()) - true_row)
        err_int = abs(float(result_int[:, 1].mean()) - true_row)
        # Sub-pixel should be at most as bad as integer
        assert err_sp <= err_int + 0.5, (
            f"Sub-pixel error {err_sp:.3f} should not be much worse than integer {err_int:.3f}"
        )

    def test_no_line_returns_empty(self) -> None:
        """Dark frame with no laser should return an empty array."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = extract_laser_line(frame, threshold=100, min_pixels=10)
        assert result.shape == (0, 2), f"Expected (0,2), got {result.shape}"

    def test_below_min_pixels_returns_empty(self) -> None:
        """A line shorter than min_pixels should not be reported."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw a 5-column green stripe in the centre
        frame[120, 160:165, 1] = 220
        result = extract_laser_line(frame, threshold=100, min_pixels=20)
        assert result.shape[0] == 0, "Short line should be rejected"

    def test_wrong_shape_logs_warning(self) -> None:
        """Grayscale input should return empty array gracefully."""
        frame = np.zeros((240, 320), dtype=np.uint8)
        result = extract_laser_line(frame, threshold=100)
        assert result.shape == (0, 2)

    def test_col_values_in_range(self) -> None:
        """All returned column indices should be within the image width."""
        frame = make_laser_frame(width=200, height=150, row=75.0)
        result = extract_laser_line(frame, threshold=80, min_pixels=5)
        if result.shape[0] > 0:
            assert result[:, 0].min() >= 0
            assert result[:, 0].max() < 200

    def test_row_values_in_range(self) -> None:
        """All returned row indices should be within the image height."""
        frame = make_laser_frame(width=200, height=150, row=75.0)
        result = extract_laser_line(frame, threshold=80, min_pixels=5)
        if result.shape[0] > 0:
            assert result[:, 1].min() >= 0
            assert result[:, 1].max() < 150

    def test_detects_vertical_line(self) -> None:
        """A near-vertical laser line should be preserved across its height."""
        true_col = 95.0
        frame = make_laser_frame_vertical(width=200, height=160, col=true_col, noise_amplitude=0)
        result = extract_laser_line(frame, threshold=80, min_pixels=20, subpixel=True)
        assert result.shape[0] >= 145, f"Expected many vertical points, got {result.shape[0]}"
        mean_col = float(result[:, 0].mean())
        assert abs(mean_col - true_col) < 2.0, (
            f"Mean detected column {mean_col:.2f} differs from truth {true_col} by > 2 px"
        )
        assert float(result[:, 1].max() - result[:, 1].min()) > 140.0

    def test_row_green_mode_tracks_vertical_line(self) -> None:
        """row_green mode should return about one point per visible row."""
        true_col = 95.25
        frame = make_laser_frame_vertical(width=200, height=160, col=true_col, noise_amplitude=0)
        result = extract_laser_line(
            frame,
            threshold=80,
            min_pixels=20,
            subpixel=True,
            mode="row_green",
        )
        assert result.shape[0] >= 145, f"Expected many rows detected, got {result.shape[0]}"
        mean_col = float(result[:, 0].mean())
        assert abs(mean_col - true_col) < 1.0, (
            f"Mean detected column {mean_col:.2f} differs from truth {true_col} by > 1 px"
        )
        assert float(result[:, 1].max() - result[:, 1].min()) > 140.0

    def test_detects_bent_polyline(self) -> None:
        """A bent laser line should not collapse to a tiny set of points."""
        frame = make_laser_frame_polyline(
            points=[(108.0, 20.0), (122.0, 80.0), (94.0, 145.0)],
            width=220,
            height=180,
            noise_amplitude=0,
        )
        result = extract_laser_line(frame, threshold=80, min_pixels=20, subpixel=True)
        assert result.shape[0] >= 100, f"Expected dense bent line, got {result.shape[0]}"
        assert float(result[:, 0].max() - result[:, 0].min()) >= 25.0
        assert float(result[:, 1].max() - result[:, 1].min()) >= 110.0

    def test_preserves_segments_across_large_gap(self) -> None:
        """Separated visible segments should both survive instead of one replacing the other."""
        frame = make_laser_frame_polyline(
            points=[(100.0, 15.0), (116.0, 70.0), (92.0, 150.0)],
            width=220,
            height=180,
            noise_amplitude=0,
            gap_segments=[(0.42, 0.58)],
        )
        result = extract_laser_line(frame, threshold=80, min_pixels=20, subpixel=True)
        assert result.shape[0] >= 80, f"Expected both visible branches to remain, got {result.shape[0]}"
        rows = np.sort(result[:, 1])
        row_gaps = np.diff(rows)
        assert row_gaps.max() > 8.0, "A large optical gap should remain visible in the extracted line"


# --------------------------------------------------------------------------- #
# triangulate
# --------------------------------------------------------------------------- #


def _make_camera_matrix(
    fx: float = 800.0, fy: float = 800.0, cx: float = 320.0, cy: float = 240.0
) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _zero_dist() -> np.ndarray:
    return np.zeros(5, dtype=np.float64)


class TestTriangulate:
    """Tests for scanner.processing.triangulation.triangulate."""

    def test_empty_input_returns_empty(self) -> None:
        """Empty pixel array should return empty 3D array."""
        result = triangulate(
            np.empty((0, 2), dtype=np.float32),
            _make_camera_matrix(),
            _zero_dist(),
            np.array([0.0, 0.0, 1.0, -300.0]),
            0.0,
        )
        assert result.shape == (0, 3)

    def test_output_shape(self) -> None:
        """Output should have shape (N, 3)."""
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
        """Points on the optical axis should intersect the frontal plane at z=300."""
        # Pixel at image centre (320, 240) → normalised coords (0, 0)
        # → camera ray [0, 0, 1]
        # Plane z = 300: [0, 0, 1, -300]
        line = np.array([[320.0, 240.0]], dtype=np.float32)
        result = triangulate(
            line,
            _make_camera_matrix(cx=320.0, cy=240.0),
            _zero_dist(),
            np.array([0.0, 0.0, 1.0, -300.0]),
            0.0,
        )
        assert result.shape == (1, 3)
        assert abs(result[0, 2] - 300.0) < 1e-6, f"Z should be 300 mm, got {result[0, 2]}"
        assert abs(result[0, 0]) < 1e-6, f"X should be 0, got {result[0, 0]}"
        assert abs(result[0, 1]) < 1e-6, f"Y should be 0, got {result[0, 1]}"

    def test_rotation_identity(self) -> None:
        """Zero rotation angle should not change X coordinate for axis-aligned rays."""
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
        """Rays parallel to the plane (dot=0) should be silently dropped."""
        # Plane normal = [0,1,0] (horizontal plane) and ray direction = [x,0,1]
        # with y_n=0 → ray dot [0,1,0] = 0 → degenerate
        # Use a pixel at row = cy so that y_n = 0
        line = np.array([[320.0, 240.0]], dtype=np.float32)
        # Plane [0,1,0,0] → y=0; ray to (0,0,1): dot=0 → no intersection
        result = triangulate(
            line,
            _make_camera_matrix(cx=320.0, cy=240.0),
            _zero_dist(),
            np.array([0.0, 1.0, 0.0, 0.0]),
            0.0,
        )
        assert result.shape[0] == 0, "Parallel ray should produce no intersection"

    def test_bad_shape_raises(self) -> None:
        """Wrong line shape should raise ValueError."""
        with pytest.raises(ValueError):
            triangulate(
                np.zeros((5, 3)),
                _make_camera_matrix(),
                _zero_dist(),
                np.array([0, 0, 1, -300.0]),
                0.0,
            )
