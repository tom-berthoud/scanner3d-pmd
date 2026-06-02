"""tests.test_reconstruction — Unit tests for point cloud assembly and filtering."""

import numpy as np
import pytest

from scanner.reconstruction import filter_outliers, fuse_half_turn_profiles, merge_profiles


# --------------------------------------------------------------------------- #
# merge_profiles
# --------------------------------------------------------------------------- #


class TestMergeProfiles:
    """Tests for scanner.reconstruction.pointcloud.merge_profiles."""

    def test_basic_merge(self) -> None:
        """Two profiles of N points each should yield 2N points."""
        a = np.random.rand(50, 3)
        b = np.random.rand(30, 3)
        result = merge_profiles([a, b])
        assert result.shape == (80, 3)

    def test_skips_empty_profiles(self) -> None:
        """Empty (0,3) arrays should be silently ignored."""
        a = np.random.rand(20, 3)
        empty = np.empty((0, 3))
        result = merge_profiles([a, empty])
        assert result.shape == (20, 3)

    def test_all_empty_returns_empty(self) -> None:
        """All-empty list should return (0,3)."""
        result = merge_profiles([np.empty((0, 3)), np.empty((0, 3))])
        assert result.shape == (0, 3)

    def test_empty_list_returns_empty(self) -> None:
        """Empty profile list should return (0,3)."""
        result = merge_profiles([])
        assert result.shape == (0, 3)

    def test_dtype_is_float64(self) -> None:
        """Result should be float64."""
        a = np.random.rand(10, 3).astype(np.float32)
        result = merge_profiles([a])
        assert result.dtype == np.float64

    def test_values_preserved(self) -> None:
        """Merged values should match the original concatenated array."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([[7.0, 8.0, 9.0]])
        result = merge_profiles([a, b])
        expected = np.vstack([a, b])
        np.testing.assert_allclose(result, expected)

    def test_single_profile(self) -> None:
        """Single profile should be returned as-is (converted to float64)."""
        a = np.ones((100, 3), dtype=np.float32)
        result = merge_profiles([a])
        assert result.shape == (100, 3)


# --------------------------------------------------------------------------- #
# fuse_half_turn_profiles
# --------------------------------------------------------------------------- #


class TestFuseHalfTurnProfiles:
    """Tests for duplicate profile fusion half a turn apart."""

    def test_averages_close_half_turn_pair(self) -> None:
        base = np.column_stack(
            (
                np.linspace(-10.0, 10.0, 20),
                np.zeros(20),
                np.full(20, 5.0),
            )
        )
        duplicate = base + np.array([0.0, 0.4, 0.0])
        profiles = [
            base,
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3)),
            duplicate,
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3)),
        ]

        fused = fuse_half_turn_profiles(
            profiles,
            n_steps=8,
            offset_tolerance_steps=0,
            max_pair_distance_mm=1.0,
            min_profile_points=4,
        )

        non_empty = [p for p in fused if p.shape[0] > 0]
        assert len(non_empty) == 1
        np.testing.assert_allclose(non_empty[0][:, 1], np.full(20, 0.2), atol=1e-6)

    def test_keeps_far_half_turn_pair(self) -> None:
        base = np.column_stack(
            (
                np.linspace(-10.0, 10.0, 20),
                np.zeros(20),
                np.full(20, 5.0),
            )
        )
        far = base + np.array([0.0, 20.0, 0.0])

        fused = fuse_half_turn_profiles(
            [base, np.empty((0, 3)), far, np.empty((0, 3))],
            n_steps=4,
            offset_tolerance_steps=0,
            max_pair_distance_mm=1.0,
            min_profile_points=4,
        )

        non_empty = [p for p in fused if p.shape[0] > 0]
        assert len(non_empty) == 2


# --------------------------------------------------------------------------- #
# filter_outliers
# --------------------------------------------------------------------------- #


def _make_sphere_cloud(n: int = 500, radius: float = 50.0, rng_seed: int = 0) -> np.ndarray:
    """Generate a roughly spherical cloud of *n* points."""
    rng = np.random.default_rng(rng_seed)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    return pts


class TestFilterOutliers:
    """Tests for scanner.reconstruction.pointcloud.filter_outliers."""

    def test_returns_subset(self) -> None:
        """Filtered cloud must be a subset of (or equal to) the input."""
        cloud = _make_sphere_cloud(300)
        filtered = filter_outliers(cloud, nb_neighbors=10, std_ratio=2.0)
        assert filtered.shape[0] <= cloud.shape[0]
        assert filtered.shape[1] == 3

    def test_removes_obvious_outliers(self) -> None:
        """Points far from the main cluster should be removed."""
        cloud = _make_sphere_cloud(200, radius=50.0)
        # Add 5 obvious outliers very far away
        outliers = np.array([[1000.0, 0.0, 0.0]] * 5)
        noisy = np.vstack([cloud, outliers])
        filtered = filter_outliers(noisy, nb_neighbors=10, std_ratio=2.0)
        # Should have removed most/all outliers
        # Check that no point is at distance > 200 from origin
        dists = np.linalg.norm(filtered, axis=1)
        assert dists.max() < 200.0, f"Outlier not removed, max dist={dists.max()}"

    def test_uniform_cloud_unchanged(self) -> None:
        """A perfectly uniform grid should lose very few points."""
        x = np.linspace(0, 10, 10)
        pts = np.array([[xi, yi, 0.0] for xi in x for yi in x])
        filtered = filter_outliers(pts, nb_neighbors=5, std_ratio=3.0)
        # Should keep at least 90% of points
        assert filtered.shape[0] >= int(0.90 * pts.shape[0])

    def test_dtype_float64(self) -> None:
        """Output dtype should always be float64."""
        cloud = _make_sphere_cloud(100)
        filtered = filter_outliers(cloud.astype(np.float32))
        assert filtered.dtype == np.float64

    def test_too_few_points_returns_unchanged(self) -> None:
        """Cloud with fewer points than nb_neighbors should be returned as-is."""
        cloud = np.random.rand(5, 3)
        filtered = filter_outliers(cloud, nb_neighbors=20)
        np.testing.assert_array_equal(filtered, cloud)

    def test_wrong_shape_raises(self) -> None:
        """Non-(N,3) array should raise ValueError."""
        with pytest.raises(ValueError):
            filter_outliers(np.random.rand(100, 2))

    def test_large_std_ratio_keeps_all(self) -> None:
        """With a very large std_ratio, no points should be removed."""
        cloud = _make_sphere_cloud(200)
        filtered = filter_outliers(cloud, nb_neighbors=10, std_ratio=1000.0)
        assert filtered.shape[0] == cloud.shape[0]
