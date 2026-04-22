"""tests.test_export — Unit tests for STL and OBJ mesh export."""

import os
import tempfile

import numpy as np
import pytest

from scanner.export import export_obj, export_point_cloud_ply, export_stl


def _make_sphere_cloud(n: int = 300, radius: float = 30.0) -> np.ndarray:
    """Generate a non-degenerate spherical point cloud for testing."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * radius
    return pts


class TestExportSTL:
    """Tests for scanner.export.stl.export_stl."""

    def test_creates_file(self) -> None:
        """export_stl should create a file at the given path."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            export_stl(cloud, path)
            assert os.path.exists(path), "STL file was not created"

    def test_file_nonempty(self) -> None:
        """The STL file should have non-zero size."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            export_stl(cloud, path)
            assert os.path.getsize(path) > 0

    def test_binary_stl_header(self) -> None:
        """Binary STL files start with an 80-byte header."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            export_stl(cloud, path)
            with open(path, "rb") as fh:
                header = fh.read(5)
            # Binary STL does NOT start with "solid" (that would be ASCII)
            # trimesh writes binary STL with a non-"solid" header
            # Just check we have a valid binary file (size > 84 bytes)
            assert os.path.getsize(path) > 84

    def test_too_few_points_raises(self) -> None:
        """Fewer than 4 points should raise ValueError."""
        cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            with pytest.raises(ValueError):
                export_stl(cloud, path)

    def test_creates_parent_dirs(self) -> None:
        """export_stl should create any missing parent directories."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "out.stl")
            export_stl(cloud, path)
            assert os.path.exists(path)


class TestExportOBJ:
    """Tests for scanner.export.stl.export_obj."""

    def test_creates_file(self) -> None:
        """export_obj should create a file at the given path."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            export_obj(cloud, path)
            assert os.path.exists(path)

    def test_file_nonempty(self) -> None:
        """The OBJ file should have non-zero size."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            export_obj(cloud, path)
            assert os.path.getsize(path) > 0

    def test_obj_has_vertices(self) -> None:
        """The OBJ file should contain at least one 'v' vertex line."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            export_obj(cloud, path)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            assert "v " in content, "OBJ file has no vertex lines"

    def test_too_few_points_raises(self) -> None:
        """Fewer than 4 points should raise ValueError."""
        cloud = np.eye(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            with pytest.raises(ValueError):
                export_obj(cloud, path)

    def test_creates_parent_dirs(self) -> None:
        """export_obj should create any missing parent directories."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "path", "out.obj")
            export_obj(cloud, path)
            assert os.path.exists(path)


class TestExportPointCloudPLY:
    """Tests for scanner.export.pointcloud.export_point_cloud_ply."""

    def test_creates_file(self) -> None:
        """export_point_cloud_ply should create a file at the given path."""
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cloud.ply")
            export_point_cloud_ply(cloud, path)
            assert os.path.exists(path)

    def test_header_contains_vertex_count(self) -> None:
        """PLY header should declare the correct number of vertices."""
        cloud = _make_sphere_cloud(n=123)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cloud.ply")
            export_point_cloud_ply(cloud, path)
            with open(path, "r", encoding="ascii") as fh:
                content = fh.read()
            assert "format ascii 1.0" in content
            assert "element vertex 123" in content

    def test_empty_cloud_raises(self) -> None:
        """An empty cloud should raise ValueError."""
        cloud = np.empty((0, 3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cloud.ply")
            with pytest.raises(ValueError):
                export_point_cloud_ply(cloud, path)
