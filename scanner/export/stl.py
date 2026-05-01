"""scanner.export.stl - Surface reconstruction and STL/OBJ export.

The scanner produces an unstructured 3-D point cloud.  Mesh generation is done
with Open3D's screened Poisson reconstruction:

1. estimate point normals from local neighbours,
2. orient normals consistently across the cloud,
3. reconstruct the surface with Poisson,
4. remove low-density Poisson artefacts,
5. write the mesh as STL or OBJ.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Sequence

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
        return self


def export_stl(
    cloud: np.ndarray,
    path: str,
    profiles: Sequence[np.ndarray] | None = None,
    mesh_mode: str | None = None,
    alpha: float | None = None,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> None:
    """Export *cloud* as a binary STL mesh reconstructed with Poisson.

    ``profiles``, ``mesh_mode`` and ``alpha`` are accepted for compatibility
    with older callers, but the mesh is always generated from the merged 3-D
    point cloud using Poisson reconstruction.
    """
    _ = profiles, mesh_mode, alpha
    _export_mesh(cloud, path, file_type="stl", poisson=poisson)


def export_obj(
    cloud: np.ndarray,
    path: str,
    profiles: Sequence[np.ndarray] | None = None,
    mesh_mode: str | None = None,
    alpha: float | None = None,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> None:
    """Export *cloud* as a Wavefront OBJ mesh reconstructed with Poisson."""
    _ = profiles, mesh_mode, alpha
    _export_mesh(cloud, path, file_type="obj", poisson=poisson)


def _export_mesh(
    cloud: np.ndarray,
    path: str,
    file_type: str,
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> None:
    mesh = _cloud_to_poisson_mesh(cloud, poisson=poisson)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    try:
        import open3d as o3d  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("open3d not available - install with: pip install open3d") from exc

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
    poisson: PoissonMeshConfig | dict[str, Any] | None = None,
) -> "o3d.geometry.TriangleMesh":  # type: ignore[name-defined]
    """Convert a point cloud to an Open3D triangle mesh using Poisson."""
    cfg = (
        poisson
        if isinstance(poisson, PoissonMeshConfig)
        else PoissonMeshConfig.from_mapping(poisson)
    )
    points = _prepare_cloud(cloud, cfg.deduplicate_decimals)

    try:
        import open3d as o3d  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("open3d not available - install with: pip install open3d") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

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

    orientation_k = min(cfg.orientation_k, points.shape[0] - 1)
    logger.info("Poisson mesh: orienting normals (k=%d)", orientation_k)
    pcd.orient_normals_consistent_tangent_plane(orientation_k)
    pcd.normalize_normals()

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

    _remove_low_density_vertices(mesh, densities, cfg.density_quantile)
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
