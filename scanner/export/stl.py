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

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "PoissonMeshConfig":
        """Build a config from settings.yaml values."""
        if not values:
            return cls()
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
