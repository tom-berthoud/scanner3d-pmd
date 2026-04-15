"""scanner.export.stl — Export point cloud as STL or OBJ mesh.

Converts a 3D point cloud to a watertight mesh using trimesh's convex hull
(fast and reliable) with an optional alpha-shape pass for concave objects.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def export_stl(cloud: np.ndarray, path: str) -> None:
    """Export *cloud* as a binary STL file at *path*.

    The mesh is constructed via convex hull if the cloud has ≥ 4 points.
    For flat or near-degenerate clouds a simple alpha-shape is attempted
    first; on failure the convex hull is used as fallback.

    Args:
        cloud: Float array of shape (N, 3) — point cloud in mm.
        path: Destination file path (must end with .stl).

    Raises:
        ValueError: if *cloud* has fewer than 4 points.
        RuntimeError: if mesh construction fails.
    """
    mesh = _cloud_to_mesh(cloud)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.export(path, file_type="stl")
    file_size = os.path.getsize(path)
    logger.info(
        "export_stl: saved %d vertices / %d faces to %s (%.1f KB)",
        len(mesh.vertices),
        len(mesh.faces),
        path,
        file_size / 1024.0,
    )


def export_obj(cloud: np.ndarray, path: str) -> None:
    """Export *cloud* as a Wavefront OBJ file at *path*.

    Args:
        cloud: Float array of shape (N, 3) — point cloud in mm.
        path: Destination file path (must end with .obj).

    Raises:
        ValueError: if *cloud* has fewer than 4 points.
        RuntimeError: if mesh construction fails.
    """
    mesh = _cloud_to_mesh(cloud)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.export(path, file_type="obj")
    file_size = os.path.getsize(path)
    logger.info(
        "export_obj: saved %d vertices / %d faces to %s (%.1f KB)",
        len(mesh.vertices),
        len(mesh.faces),
        path,
        file_size / 1024.0,
    )


def _cloud_to_mesh(cloud: np.ndarray) -> "trimesh.Trimesh":  # type: ignore[name-defined]
    """Convert a point cloud to a trimesh mesh.

    Attempts alpha shape first; falls back to convex hull.

    Args:
        cloud: Float array (N, 3).

    Returns:
        A trimesh.Trimesh instance.

    Raises:
        ValueError: if *cloud* has fewer than 4 non-degenerate points.
    """
    try:
        import trimesh  # type: ignore[import]
        import trimesh.creation  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("trimesh not available — install with: pip install trimesh") from exc

    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {cloud.shape}")

    n = cloud.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 points to build a mesh, got {n}")

    # Remove duplicate points
    cloud_unique = np.unique(cloud.round(decimals=4), axis=0)
    if cloud_unique.shape[0] < 4:
        raise ValueError("Too few unique points after deduplication")

    # Try alpha shape (better for concave objects)
    mesh = None
    try:
        import trimesh.creation  # noqa: F811

        # Compute a reasonable alpha value from the point cloud bounding box
        bbox_diag = float(np.linalg.norm(cloud_unique.max(axis=0) - cloud_unique.min(axis=0)))
        alpha = bbox_diag / 10.0

        alpha_mesh = trimesh.creation.icosphere()  # placeholder
        # Use trimesh's alpha_shape if available
        if hasattr(trimesh, "creation") and hasattr(trimesh.creation, "alpha_shape"):
            alpha_mesh = trimesh.creation.alpha_shape(cloud_unique, alpha=alpha)
            if alpha_mesh is not None and len(alpha_mesh.faces) > 0 and alpha_mesh.is_watertight:
                mesh = alpha_mesh
                logger.debug("Mesh built via alpha shape (alpha=%.2f)", alpha)
    except Exception as alpha_exc:
        logger.debug("Alpha shape failed: %s — falling back to convex hull", alpha_exc)

    # Fallback: convex hull (always valid for ≥ 4 non-coplanar points)
    if mesh is None:
        try:
            hull = trimesh.convex.convex_hull(cloud_unique)
            mesh = hull
            logger.debug("Mesh built via convex hull (%d faces)", len(mesh.faces))
        except Exception as hull_exc:
            raise RuntimeError(f"Mesh construction failed: {hull_exc}") from hull_exc

    return mesh
