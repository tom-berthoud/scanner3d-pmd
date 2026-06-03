"""scanner.export.stl - Poisson surface reconstruction and STL/OBJ export."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import os
from types import ModuleType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoissonMeshConfig:
    """Tunable parameters for Open3D Poisson surface reconstruction."""

    normal_radius_mm: float = 8.0
    normal_max_nn: int = 30
    orientation_k: int = 50
    depth: int = 8
    width: float = 0.0
    scale: float = 1.1
    linear_fit: bool = False
    density_quantile: float = 0.02
    deduplicate_decimals: int = 4
    close_horizontal_holes: bool = True
    horizontal_normal_tolerance: float = 0.85
    mesh_clip_plane: tuple[float, float, float, float] | None = None
    mesh_clip_margin_mm: float = 0.0
    mesh_clip_cap: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "PoissonMeshConfig":
        """Build a config from settings.yaml values."""
        if not values:
            return cls()
        clip_plane_raw = values.get("mesh_clip_plane")
        clip_plane = None
        if clip_plane_raw is not None:
            clip_values = tuple(float(v) for v in clip_plane_raw)
            if len(clip_values) != 4:
                raise ValueError("poisson.mesh_clip_plane must contain 4 values [a,b,c,d]")
            clip_plane = clip_values

        return cls(
            normal_radius_mm=float(values.get("normal_radius_mm", cls.normal_radius_mm)),
            normal_max_nn=int(values.get("normal_max_nn", cls.normal_max_nn)),
            orientation_k=int(values.get("orientation_k", cls.orientation_k)),
            depth=int(values.get("depth", cls.depth)),
            width=float(values.get("width", cls.width)),
            scale=float(values.get("scale", cls.scale)),
            linear_fit=bool(values.get("linear_fit", cls.linear_fit)),
            density_quantile=float(values.get("density_quantile", cls.density_quantile)),
            deduplicate_decimals=int(
                values.get("deduplicate_decimals", cls.deduplicate_decimals)
            ),
            close_horizontal_holes=bool(
                values.get("close_horizontal_holes", cls.close_horizontal_holes)
            ),
            horizontal_normal_tolerance=float(
                values.get(
                    "horizontal_normal_tolerance",
                    cls.horizontal_normal_tolerance,
                )
            ),
            mesh_clip_plane=clip_plane,
            mesh_clip_margin_mm=float(
                values.get("mesh_clip_margin_mm", cls.mesh_clip_margin_mm)
            ),
            mesh_clip_cap=bool(values.get("mesh_clip_cap", cls.mesh_clip_cap)),
        ).validated()

    def validated(self) -> "PoissonMeshConfig":
        """Return self after checking parameter ranges."""
        if self.normal_radius_mm <= 0.0:
            raise ValueError("poisson.normal_radius_mm must be > 0")
        if self.normal_max_nn < 3:
            raise ValueError("poisson.normal_max_nn must be >= 3")
        if self.orientation_k < 3:
            raise ValueError("poisson.orientation_k must be >= 3")
        if self.depth < 4:
            raise ValueError("poisson.depth must be >= 4")
        if self.width < 0.0:
            raise ValueError("poisson.width must be >= 0")
        if self.scale <= 1.0:
            raise ValueError("poisson.scale must be > 1")
        if not 0.0 <= self.density_quantile < 1.0:
            raise ValueError("poisson.density_quantile must be in [0, 1)")
        if self.deduplicate_decimals < 0:
            raise ValueError("poisson.deduplicate_decimals must be >= 0")
        if not 0.0 <= self.horizontal_normal_tolerance <= 1.0:
            raise ValueError("poisson.horizontal_normal_tolerance must be in [0, 1]")
        if self.mesh_clip_margin_mm < 0.0:
            raise ValueError("poisson.mesh_clip_margin_mm must be >= 0")
        return self


def export_stl(
    cloud: np.ndarray,
    path: str,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> None:
    """Export *cloud* as a binary STL mesh reconstructed with Poisson."""
    _export_mesh(cloud, path, file_type="stl", poisson=poisson)


def export_obj(
    cloud: np.ndarray,
    path: str,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> None:
    """Export *cloud* as a Wavefront OBJ mesh reconstructed with Poisson."""
    _export_mesh(cloud, path, file_type="obj", poisson=poisson)


def _export_mesh(
    cloud: np.ndarray,
    path: str,
    file_type: str,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> None:
    mesh = _cloud_to_poisson_mesh(cloud, poisson=poisson)
    o3d = _require_open3d()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    ok = o3d.io.write_triangle_mesh(
        path,
        mesh,
        write_ascii=(file_type == "obj"),
        compressed=False,
        write_vertex_normals=True,
        write_triangle_uvs=False,
    )
    if not ok:
        raise RuntimeError(f"Open3D failed to write {file_type.upper()} mesh to {path}")

    file_size = os.path.getsize(path)
    logger.info(
        "export_%s: saved %d vertices / %d faces to %s (%.1f KB)",
        file_type,
        len(mesh.vertices),
        len(mesh.triangles),
        path,
        file_size / 1024.0,
    )


def _cloud_to_poisson_mesh(
    cloud: np.ndarray,
    o3d: ModuleType | None = None,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> "o3d.geometry.TriangleMesh":  # type: ignore[name-defined]
    """Convert a 3-D point cloud to an Open3D triangle mesh using Poisson."""
    cfg = (
        poisson
        if isinstance(poisson, PoissonMeshConfig)
        else PoissonMeshConfig.from_mapping(poisson)
    )
    points = _prepare_cloud(cloud, cfg.deduplicate_decimals)
    o3d = o3d or _require_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _estimate_poisson_normals(pcd, point_count=points.shape[0], cfg=cfg, o3d=o3d)
    mesh, densities = _run_poisson_reconstruction(pcd, cfg=cfg, o3d=o3d)

    _remove_low_density_vertices(mesh, densities, cfg.density_quantile)
    _clean_mesh(mesh)
    if cfg.close_horizontal_holes:
        _cap_horizontal_boundary_loops(
            mesh,
            normal_tolerance=cfg.horizontal_normal_tolerance,
        )
    if cfg.mesh_clip_plane is not None:
        _clip_mesh_by_plane(
            mesh,
            np.asarray(cfg.mesh_clip_plane, dtype=np.float64),
            margin_mm=cfg.mesh_clip_margin_mm,
            cap=cfg.mesh_clip_cap,
        )
        _clean_mesh(mesh)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Poisson reconstruction produced an empty mesh")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    logger.info(
        "Poisson mesh: generated %d vertices / %d faces",
        len(mesh.vertices),
        len(mesh.triangles),
    )
    return mesh


def _require_open3d() -> ModuleType:
    try:
        import open3d as o3d  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for Poisson mesh export. Install it with: pip install open3d"
        ) from exc
    return o3d


def _estimate_poisson_normals(
    pcd: Any,
    point_count: int,
    cfg: PoissonMeshConfig,
    o3d: ModuleType,
) -> None:
    logger.info(
        "Poisson mesh: estimating normals (radius=%.2f mm, max_nn=%d)",
        cfg.normal_radius_mm,
        cfg.normal_max_nn,
    )
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius_mm,
            max_nn=cfg.normal_max_nn,
        )
    )

    orientation_k = min(cfg.orientation_k, point_count - 1)
    logger.info("Poisson mesh: orienting normals (k=%d)", orientation_k)
    pcd.orient_normals_consistent_tangent_plane(orientation_k)
    pcd.normalize_normals()


def _run_poisson_reconstruction(
    pcd: Any,
    cfg: PoissonMeshConfig,
    o3d: ModuleType,
) -> tuple[Any, Any]:
    logger.info(
        "Poisson mesh: reconstructing surface (depth=%d, scale=%.2f, linear_fit=%s)",
        cfg.depth,
        cfg.scale,
        cfg.linear_fit,
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=cfg.depth,
        width=cfg.width,
        scale=cfg.scale,
        linear_fit=cfg.linear_fit,
    )
    return mesh, densities


def _prepare_cloud(cloud: np.ndarray, deduplicate_decimals: int) -> np.ndarray:
    points = np.asarray(cloud, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {points.shape}")

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.shape[0] < 4:
        raise ValueError(f"Need at least 4 finite points to build a mesh, got {points.shape[0]}")

    points = np.unique(points.round(decimals=deduplicate_decimals), axis=0)
    if points.shape[0] < 4:
        raise ValueError("Too few unique points after deduplication")

    return points


def _remove_low_density_vertices(
    mesh: "o3d.geometry.TriangleMesh",  # type: ignore[name-defined]
    densities: "o3d.utility.DoubleVector",  # type: ignore[name-defined]
    density_quantile: float,
) -> None:
    if density_quantile <= 0.0:
        return

    density_values = np.asarray(densities)
    if density_values.size == 0:
        return

    threshold = float(np.quantile(density_values, density_quantile))
    low_density_mask = density_values < threshold
    removed = int(low_density_mask.sum())
    if removed > 0:
        mesh.remove_vertices_by_mask(low_density_mask)
        logger.info(
            "Poisson mesh: removed %d low-density vertices (quantile=%.3f)",
            removed,
            density_quantile,
        )


def _clean_mesh(mesh: "o3d.geometry.TriangleMesh") -> None:  # type: ignore[name-defined]
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()


def _cap_horizontal_boundary_loops(
    mesh: "o3d.geometry.TriangleMesh",  # type: ignore[name-defined]
    normal_tolerance: float = 0.85,
) -> int:
    """Fill open boundary loops whose best-fit plane is horizontal.

    The scanner uses Y as the vertical axis. This intentionally only closes
    nearly horizontal holes, such as missing top/bottom caps on a cylinder,
    without trying to invent arbitrary side geometry.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if vertices.size == 0 or triangles.size == 0:
        return 0

    boundary_components = _find_boundary_components(triangles)
    if not boundary_components:
        return 0

    new_vertices = vertices.tolist()
    new_triangles = triangles.tolist()
    caps_added = 0
    vertical_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for component in boundary_components:
        if len(component) < 3:
            continue

        component_indices = np.array(sorted(component), dtype=np.int64)
        points = vertices[component_indices]
        centroid = points.mean(axis=0)
        centered = points - centroid
        if np.linalg.matrix_rank(centered) < 2:
            continue

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        if abs(float(np.dot(normal, vertical_axis))) < normal_tolerance:
            continue

        ordered_indices = _order_horizontal_loop_vertices(component_indices, points, centroid)
        if ordered_indices.size < 3:
            continue

        center_idx = len(new_vertices)
        new_vertices.append(centroid.tolist())

        upward = centroid[1] >= vertices[:, 1].mean()
        for i, v0 in enumerate(ordered_indices):
            v1 = ordered_indices[(i + 1) % ordered_indices.size]
            if upward:
                new_triangles.append([int(center_idx), int(v0), int(v1)])
            else:
                new_triangles.append([int(center_idx), int(v1), int(v0)])
        caps_added += 1

    if caps_added == 0:
        return 0

    o3d = _require_open3d()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(new_vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(new_triangles, dtype=np.int32))
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    logger.info("Poisson mesh: capped %d horizontal boundary loop(s)", caps_added)
    return caps_added


def _clip_mesh_by_plane(
    mesh: "o3d.geometry.TriangleMesh",  # type: ignore[name-defined]
    plane: np.ndarray,
    *,
    margin_mm: float = 0.0,
    cap: bool = True,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        return

    normal = np.asarray(plane[:3], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return
    normal = normal / norm
    d = float(plane[3]) / norm - float(margin_mm)
    signed = vertices @ normal + d

    new_vertices = vertices.tolist()
    new_triangles: list[list[int]] = []
    intersection_cache: dict[tuple[int, int], int] = {}
    cut_segments: list[tuple[int, int]] = []

    def _intersection_idx(i0: int, i1: int) -> int:
        key = tuple(sorted((int(i0), int(i1))))
        cached = intersection_cache.get(key)
        if cached is not None:
            return cached
        s0 = float(signed[i0])
        s1 = float(signed[i1])
        t = s0 / (s0 - s1)
        point = vertices[i0] + t * (vertices[i1] - vertices[i0])
        idx = len(new_vertices)
        new_vertices.append(point.tolist())
        intersection_cache[key] = idx
        return idx

    for tri in triangles:
        poly = [int(tri[0]), int(tri[1]), int(tri[2])]
        clipped: list[int] = []
        intersections: list[int] = []
        for pos, current in enumerate(poly):
            previous = poly[pos - 1]
            current_inside = signed[current] <= 1e-9
            previous_inside = signed[previous] <= 1e-9
            if current_inside != previous_inside:
                inter = _intersection_idx(previous, current)
                clipped.append(inter)
                intersections.append(inter)
            if current_inside:
                clipped.append(current)

        if len(clipped) < 3:
            continue
        for k in range(1, len(clipped) - 1):
            new_triangles.append([clipped[0], clipped[k], clipped[k + 1]])
        if cap and len(intersections) == 2 and intersections[0] != intersections[1]:
            cut_segments.append((intersections[0], intersections[1]))

    if cap and cut_segments:
        new_triangles.extend(_triangulate_cut_segments(new_vertices, cut_segments, normal))

    o3d = _require_open3d()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(new_vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(new_triangles, dtype=np.int32))
    logger.info(
        "Poisson mesh: plane-clipped to %d vertices / %d faces (cap=%s)",
        len(new_vertices),
        len(new_triangles),
        cap,
    )


def _triangulate_cut_segments(
    vertices: list[list[float]],
    segments: list[tuple[int, int]],
    normal: np.ndarray,
) -> list[list[int]]:
    unique_edges = {tuple(sorted((int(a), int(b)))) for a, b in segments if a != b}
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in unique_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    loops: list[list[int]] = []
    used_edges: set[tuple[int, int]] = set()
    for edge in unique_edges:
        if edge in used_edges:
            continue
        start, nxt = edge
        loop = [start]
        prev = start
        current = nxt
        used_edges.add(edge)
        while True:
            loop.append(current)
            neighbors = adjacency.get(current, [])
            candidates = [n for n in neighbors if n != prev]
            if not candidates:
                break
            nxt = candidates[0]
            edge_key = tuple(sorted((current, nxt)))
            if nxt == start:
                used_edges.add(edge_key)
                break
            if edge_key in used_edges:
                break
            used_edges.add(edge_key)
            prev, current = current, nxt
        if len(loop) >= 3 and loop[0] != loop[-1]:
            loops.append(loop)

    if not loops:
        return []

    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, n))) > 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.cross(helper, n)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)

    triangles: list[list[int]] = []
    vertex_arr = np.asarray(vertices, dtype=np.float64)
    for loop in loops:
        points_3d = vertex_arr[np.asarray(loop, dtype=np.int64)]
        points_2d = np.column_stack((points_3d @ u, points_3d @ v))
        tris = _ear_clip_loop(points_2d)
        for a, b, c in tris:
            triangles.append([int(loop[a]), int(loop[b]), int(loop[c])])
    return triangles


def _ear_clip_loop(points: np.ndarray) -> list[tuple[int, int, int]]:
    n = points.shape[0]
    if n < 3:
        return []
    order = list(range(n))
    if _polygon_area(points) < 0:
        order.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(order) > 3 and guard < n * n:
        guard += 1
        clipped = False
        for idx in range(len(order)):
            prev_i = order[idx - 1]
            curr_i = order[idx]
            next_i = order[(idx + 1) % len(order)]
            if not _is_convex(points[prev_i], points[curr_i], points[next_i]):
                continue
            tri = (points[prev_i], points[curr_i], points[next_i])
            if any(
                _point_in_triangle(points[other], tri)
                for other in order
                if other not in (prev_i, curr_i, next_i)
            ):
                continue
            triangles.append((prev_i, curr_i, next_i))
            del order[idx]
            clipped = True
            break
        if not clipped:
            break

    if len(order) == 3:
        triangles.append((order[0], order[1], order[2]))
    return triangles


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _is_convex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    ab = b - a
    bc = c - b
    return float(ab[0] * bc[1] - ab[1] * bc[0]) > 1e-10


def _point_in_triangle(p: np.ndarray, tri: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    a, b, c = tri
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= 1e-12:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-10 and v >= -1e-10 and (u + v) <= 1.0 + 1e-10


def _find_boundary_components(triangles: np.ndarray) -> list[set[int]]:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            edge_counts[tuple(sorted((u, v)))] += 1

    adjacency: dict[int, set[int]] = defaultdict(set)
    for (u, v), count in edge_counts.items():
        if count == 1:
            adjacency[u].add(v)
            adjacency[v].add(u)

    components: list[set[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        component: set[int] = set()
        visited.add(start)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _order_horizontal_loop_vertices(
    indices: np.ndarray,
    points: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    offsets = points - centroid
    angles = np.arctan2(offsets[:, 2], offsets[:, 0])
    return indices[np.argsort(angles)]
