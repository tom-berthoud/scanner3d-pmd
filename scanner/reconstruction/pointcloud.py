"""scanner.reconstruction.pointcloud — 3D point cloud operations.

Provides utilities to merge multiple scan profiles into a single point cloud
and to remove statistical outliers using a KD-tree based filter.
"""

import logging

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def merge_profiles(profiles: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of per-step 3D profiles into a single point cloud.

    Empty profiles (0 points) are silently skipped.

    Args:
        profiles: List of (N_i, 3) float arrays, one per rotation step.

    Returns:
        Float64 array of shape (M, 3) with all valid points concatenated.
        Returns an empty (0, 3) array if all profiles are empty.
    """
    non_empty = [p for p in profiles if p.ndim == 2 and p.shape[0] > 0 and p.shape[1] == 3]
    if not non_empty:
        logger.warning("merge_profiles: no valid profiles to merge")
        return np.empty((0, 3), dtype=np.float64)

    cloud = np.vstack(non_empty).astype(np.float64)
    logger.info(
        "merge_profiles: merged %d profiles → %d points total", len(non_empty), len(cloud)
    )
    return cloud


def filter_outliers(
    cloud: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers from a point cloud using a KD-tree.

    For each point, the mean distance to its *nb_neighbors* nearest
    neighbours is computed.  Points whose mean distance exceeds
    (global_mean + std_ratio * global_std) are removed.

    Args:
        cloud: Float array of shape (N, 3).
        nb_neighbors: Number of nearest neighbours to consider (default 20).
        std_ratio: Points with mean-neighbour-distance > mean + std_ratio*std
            are treated as outliers (default 2.0).

    Returns:
        Filtered float64 array of shape (M, 3) with M ≤ N.

    Raises:
        ValueError: if *cloud* is not a 2-D (N, 3) array.
    """
    from scipy.spatial import KDTree  # type: ignore[import]

    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {cloud.shape}")

    n = cloud.shape[0]
    if n <= nb_neighbors:
        logger.warning(
            "filter_outliers: point count (%d) ≤ nb_neighbors (%d) — returning unchanged",
            n,
            nb_neighbors,
        )
        return cloud.copy()

    tree = KDTree(cloud)
    # Query k+1 neighbours because the point itself is included at distance 0
    distances, _ = tree.query(cloud, k=nb_neighbors + 1)
    # Exclude the self-distance (index 0 = 0.0)
    mean_distances = distances[:, 1:].mean(axis=1)  # (N,)

    global_mean = float(mean_distances.mean())
    global_std = float(mean_distances.std())
    threshold = global_mean + std_ratio * global_std

    inlier_mask = mean_distances <= threshold
    filtered = cloud[inlier_mask]

    n_removed = n - int(inlier_mask.sum())
    logger.info(
        "filter_outliers: removed %d/%d points (threshold=%.4f mm, "
        "mean=%.4f mm, std=%.4f mm)",
        n_removed,
        n,
        threshold,
        global_mean,
        global_std,
    )
    return filtered.astype(np.float64)


def _robust_xy_bounds(points: np.ndarray, pad_mm: float) -> tuple[float, float, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    x0, x1 = float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))
    y0, y1 = float(np.quantile(y, 0.01)), float(np.quantile(y, 0.99))
    return x0 - pad_mm, x1 + pad_mm, y0 - pad_mm, y1 + pad_mm


def _points_in_polygon(x: np.ndarray, y: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon."""
    inside = np.zeros_like(x, dtype=bool)
    n = poly_xy.shape[0]
    xj, yj = poly_xy[-1, 0], poly_xy[-1, 1]
    for i in range(n):
        xi, yi = poly_xy[i, 0], poly_xy[i, 1]
        cond = ((yi > y) != (yj > y))
        denom = (yj - yi)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        x_inter = (xj - xi) * (y - yi) / denom + xi
        inside ^= cond & (x < x_inter)
        xj, yj = xi, yi
    return inside


def _compute_local_basis(points: np.ndarray, axis_mode: str, axis_index: int) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    if axis_mode == "world":
        return center, np.eye(3, dtype=np.float64)

    if axis_index not in (0, 1, 2):
        raise ValueError(f"axis_index must be 0, 1 or 2, got {axis_index}")
    cov = np.cov((points - center).T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    cols = [0, 1, 2]
    cols.remove(axis_index)
    cols.append(axis_index)
    basis = evecs[:, cols]
    if np.linalg.det(basis) < 0:
        basis[:, 1] *= -1.0
    return center, basis


def _add_flat_caps_local(
    points_local: np.ndarray,
    grid_mm: float,
    top_quantile: float,
    bottom_quantile: float,
    border_pad_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if points_local.shape[0] < 50:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    x0, x1, y0, y1 = _robust_xy_bounds(points_local, border_pad_mm)
    xs = np.arange(x0, x1 + grid_mm, grid_mm, dtype=np.float64)
    ys = np.arange(y0, y1 + grid_mm, grid_mm, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")

    xy = points_local[:, :2]
    if xy.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    hull = ConvexHull(xy)
    poly = xy[hull.vertices]
    mask = _points_in_polygon(gx.ravel(), gy.ravel(), poly).reshape(gx.shape)
    count = int(mask.sum())
    if count == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    z_top = float(np.quantile(points_local[:, 2], top_quantile))
    z_bottom = float(np.quantile(points_local[:, 2], bottom_quantile))
    top_pts = np.column_stack((gx[mask], gy[mask], np.full(count, z_top, dtype=np.float64)))
    bot_pts = np.column_stack((gx[mask], gy[mask], np.full(count, z_bottom, dtype=np.float64)))
    return top_pts.astype(np.float64), bot_pts.astype(np.float64)


def add_flat_caps_aligned(
    cloud: np.ndarray,
    *,
    enabled: bool = False,
    axis_mode: str = "pca",
    axis_index: int = 2,
    grid_mm: float = 0.8,
    top_quantile: float = 0.99,
    bottom_quantile: float = 0.01,
    border_pad_mm: float = 1.0,
) -> np.ndarray:
    """Add flat top/bottom cap points aligned to the cloud frame.

    Mirrors the successful local workflow used in reconstruction experiments.
    """
    if not enabled:
        return cloud
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {cloud.shape}")
    if cloud.shape[0] < 100:
        logger.warning("add_flat_caps_aligned: too few points (%d), skip", cloud.shape[0])
        return cloud
    if grid_mm <= 0:
        raise ValueError(f"grid_mm must be > 0, got {grid_mm}")

    center, basis = _compute_local_basis(cloud, axis_mode=axis_mode, axis_index=axis_index)
    local = (cloud - center) @ basis
    top_l, bot_l = _add_flat_caps_local(
        local,
        grid_mm=grid_mm,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        border_pad_mm=border_pad_mm,
    )
    if top_l.shape[0] == 0 and bot_l.shape[0] == 0:
        logger.warning("add_flat_caps_aligned: no cap points generated")
        return cloud

    top_w = top_l @ basis.T + center
    bot_w = bot_l @ basis.T + center
    merged = np.vstack((cloud, top_w, bot_w)).astype(np.float64)
    logger.info(
        "add_flat_caps_aligned: added top=%d, bottom=%d points (axis_mode=%s, axis_index=%d)",
        top_w.shape[0],
        bot_w.shape[0],
        axis_mode,
        axis_index,
    )
    return merged
