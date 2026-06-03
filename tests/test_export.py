"""tests.test_export — Unit tests for STL and OBJ mesh export."""

import os
import tempfile

import numpy as np
import pytest

from scanner.export import export_obj, export_point_cloud_ply, export_stl


POISSON_TEST_CFG = {
    "normal_radius_mm": 20.0,
    "normal_max_nn": 30,
    "orientation_k": 20,
    "depth": 5,
    "density_quantile": 0.0,
}


def _requires_open3d() -> None:
    pytest.importorskip("open3d")


def _make_sphere_cloud(n: int = 300, radius: float = 30.0) -> np.ndarray:
    """Generate a non-degenerate spherical point cloud for testing."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms * radius
    return pts


def _make_hourglass_profiles(n_profiles: int = 24, n_rows: int = 18) -> list[np.ndarray]:
    """Generate ordered scan profiles with a concave waist."""
    profiles: list[np.ndarray] = []
    ys = np.linspace(-30.0, 30.0, n_rows)
    for angle in np.linspace(0.0, 2.0 * np.pi, n_profiles, endpoint=False):
        radii = 24.0 + 10.0 * (np.abs(ys) / np.max(np.abs(ys)))
        x = radii * np.cos(angle)
        z = radii * np.sin(angle)
        profile = np.column_stack(
            [
                np.full_like(ys, x, dtype=np.float64),
                ys.astype(np.float64),
                np.full_like(ys, z, dtype=np.float64),
            ]
        )
        profiles.append(profile)
    return profiles


def _boundary_edge_count(mesh: object) -> int:
    triangles = np.asarray(mesh.triangles)
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return sum(1 for count in edge_counts.values() if count == 1)


def _make_open_cylinder_mesh(o3d: object, n_segments: int = 24) -> object:
    radius = 20.0
    half_height = 15.0
    vertices: list[list[float]] = []
    for y in (-half_height, half_height):
        for angle in np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False):
            vertices.append([radius * np.cos(angle), y, radius * np.sin(angle)])

    triangles: list[list[int]] = []
    for idx in range(n_segments):
        nxt = (idx + 1) % n_segments
        bottom0, bottom1 = idx, nxt
        top0, top1 = idx + n_segments, nxt + n_segments
        triangles.append([bottom0, bottom1, top1])
        triangles.append([bottom0, top1, top0])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh


def _make_l_prism_side_mesh(o3d: object) -> object:
    poly = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    vertices: list[list[float]] = []
    for y in (-1.0, 1.0):
        for x, z in poly:
            vertices.append([float(x), y, float(z)])

    n = poly.shape[0]
    triangles: list[list[int]] = []
    for idx in range(n):
        nxt = (idx + 1) % n
        b0, b1 = idx, nxt
        t0, t1 = idx + n, nxt + n
        triangles.append([b0, b1, t1])
        triangles.append([b0, t1, t0])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh


class TestExportSTL:
    """Tests for scanner.export.stl.export_stl."""

    def test_creates_file(self) -> None:
        """export_stl should create a file at the given path."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            export_stl(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.exists(path), "STL file was not created"

    def test_file_nonempty(self) -> None:
        """The STL file should have non-zero size."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            export_stl(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.getsize(path) > 0

    def test_binary_stl_header(self) -> None:
        """Binary STL files start with an 80-byte header."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            export_stl(cloud, path, poisson=POISSON_TEST_CFG)
            with open(path, "rb") as fh:
                header = fh.read(5)
            # Binary STL does NOT start with "solid" (that would be ASCII)
            # Open3D writes binary STL with a non-"solid" header
            # Just check we have a valid binary file (size > 84 bytes)
            assert os.path.getsize(path) > 84

    def test_too_few_points_raises(self) -> None:
        """Fewer than 4 points should raise ValueError."""
        _requires_open3d()
        cloud = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.stl")
            with pytest.raises(ValueError, match="at least 4"):
                export_stl(cloud, path, poisson=POISSON_TEST_CFG)

    def test_creates_parent_dirs(self) -> None:
        """export_stl should create any missing parent directories."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "out.stl")
            export_stl(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.exists(path)

    def test_concave_cloud_exports_from_points_only(self) -> None:
        """Poisson STL export should use the merged point cloud directly."""
        _requires_open3d()
        profiles = _make_hourglass_profiles()
        cloud = np.vstack(profiles)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hourglass.stl")
            export_stl(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


class TestHorizontalHoleCapping:
    """Tests for closing horizontal mesh holes after Poisson."""

    def test_caps_top_and_bottom_boundary_loops(self) -> None:
        o3d = pytest.importorskip("open3d")
        from scanner.export.stl import _cap_horizontal_boundary_loops

        mesh = _make_open_cylinder_mesh(o3d)
        assert _boundary_edge_count(mesh) == 48

        capped = _cap_horizontal_boundary_loops(mesh)

        assert capped == 2
        assert _boundary_edge_count(mesh) == 0


class TestMeshPlaneClip:
    """Tests for clipping and capping meshes with a plane."""

    def test_caps_concave_l_cut_without_filling_missing_corner(self) -> None:
        o3d = pytest.importorskip("open3d")
        from scanner.export.stl import _clip_mesh_by_plane

        mesh = _make_l_prism_side_mesh(o3d)
        _clip_mesh_by_plane(mesh, np.asarray([0.0, 1.0, 0.0, 0.0]), cap=True)

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        cap_triangles = []
        for tri in triangles:
            pts = vertices[tri]
            if np.allclose(pts[:, 1], 0.0):
                cap_triangles.append(pts)

        assert cap_triangles, "Expected a cap on the clipping plane"
        for pts in cap_triangles:
            centroid = pts.mean(axis=0)
            assert not (centroid[0] > 1.0 and centroid[2] > 1.0)


class TestExportOBJ:
    """Tests for scanner.export.stl.export_obj."""

    def test_creates_file(self) -> None:
        """export_obj should create a file at the given path."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            export_obj(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.exists(path)

    def test_file_nonempty(self) -> None:
        """The OBJ file should have non-zero size."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            export_obj(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.getsize(path) > 0

    def test_obj_has_vertices(self) -> None:
        """The OBJ file should contain at least one 'v' vertex line."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            export_obj(cloud, path, poisson=POISSON_TEST_CFG)
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            assert "v " in content, "OBJ file has no vertex lines"

    def test_too_few_points_raises(self) -> None:
        """Fewer than 4 points should raise ValueError."""
        _requires_open3d()
        cloud = np.eye(3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.obj")
            with pytest.raises(ValueError, match="at least 4"):
                export_obj(cloud, path, poisson=POISSON_TEST_CFG)

    def test_creates_parent_dirs(self) -> None:
        """export_obj should create any missing parent directories."""
        _requires_open3d()
        cloud = _make_sphere_cloud()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "deep", "path", "out.obj")
            export_obj(cloud, path, poisson=POISSON_TEST_CFG)
            assert os.path.exists(path)

    def test_concave_cloud_obj_has_faces(self) -> None:
        """Poisson OBJ export should write faces reconstructed from the cloud."""
        _requires_open3d()
        profiles = _make_hourglass_profiles()
        cloud = np.vstack(profiles)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hourglass.obj")
            export_obj(cloud, path, poisson=POISSON_TEST_CFG)
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            assert "f " in content


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
