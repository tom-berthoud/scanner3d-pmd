"""scanner.reconstruction.pointcloud — 3D point cloud operations.

Provides utilities to merge multiple scan profiles into a single point cloud
and to remove statistical outliers using a KD-tree based filter.
"""

import logging

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

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


def _profile_distance_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Return a robust symmetric nearest-neighbour distance between profiles."""
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_ba, _ = tree_a.query(b, k=1)
    d_ab, _ = tree_b.query(a, k=1)
    return float(0.5 * (np.median(d_ab) + np.median(d_ba)))


def _average_profiles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Average two nearby profile curves after resampling along their main axis."""
    combined = np.vstack((a, b)).astype(np.float64)
    center = combined.mean(axis=0)
    if combined.shape[0] < 3:
        n = min(a.shape[0], b.shape[0])
        return 0.5 * (a[:n] + b[:n])

    cov = np.cov((combined - center).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]

    def _sort_unique(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = (profile - center) @ axis
        order = np.argsort(t)
        t = t[order]
        p = profile[order]
        unique = np.concatenate(([True], np.diff(t) > 1e-9))
        return t[unique], p[unique]

    ta, pa = _sort_unique(a)
    tb, pb = _sort_unique(b)
    t0 = max(float(ta.min()), float(tb.min()))
    t1 = min(float(ta.max()), float(tb.max()))
    if t1 <= t0:
        n = min(a.shape[0], b.shape[0])
        return 0.5 * (a[:n] + b[:n])

    n_out = max(2, max(a.shape[0], b.shape[0]))
    t_grid = np.linspace(t0, t1, n_out, dtype=np.float64)
    ai = np.column_stack([np.interp(t_grid, ta, pa[:, k]) for k in range(3)])
    bi = np.column_stack([np.interp(t_grid, tb, pb[:, k]) for k in range(3)])
    return (0.5 * (ai + bi)).astype(np.float64)


def fuse_half_turn_profiles(
    profiles: list[np.ndarray],
    *,
    n_steps: int,
    enabled: bool = True,
    offset_tolerance_steps: int = 1,
    max_pair_distance_mm: float = 6.0,
    min_profile_points: int = 8,
) -> list[np.ndarray]:
    """Fuse duplicate profiles observed again about half a turn later.

    Some top-facing surfaces can be observed at step ``i`` and again near
    ``i + n_steps/2``. If both triangulated profiles are close in object space,
    this returns one averaged profile and removes the duplicate observation.
    """
    if not enabled or n_steps < 2 or not profiles:
        return profiles
    if max_pair_distance_mm <= 0:
        raise ValueError(f"max_pair_distance_mm must be > 0, got {max_pair_distance_mm}")

    total = len(profiles)
    half_turn = max(1, int(round(float(n_steps) / 2.0)))
    tolerance = max(0, int(offset_tolerance_steps))
    used: set[int] = set()
    fused: list[np.ndarray] = []
    n_pairs = 0

    for i, profile in enumerate(profiles):
        if i in used:
            continue
        if profile.ndim != 2 or profile.shape[1] != 3 or profile.shape[0] < min_profile_points:
            fused.append(profile)
            used.add(i)
            continue

        best_j: int | None = None
        best_dist = float("inf")
        for delta in range(-tolerance, tolerance + 1):
            j = i + half_turn + delta
            if j >= total or j in used:
                continue
            other = profiles[j]
            if other.ndim != 2 or other.shape[1] != 3 or other.shape[0] < min_profile_points:
                continue
            dist = _profile_distance_mm(profile.astype(np.float64), other.astype(np.float64))
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None and best_dist <= max_pair_distance_mm:
            fused.append(_average_profiles(profile.astype(np.float64), profiles[best_j].astype(np.float64)))
            used.add(i)
            used.add(best_j)
            n_pairs += 1
            continue

        fused.append(profile)
        used.add(i)

    logger.info(
        "fuse_half_turn_profiles: fused %d half-turn pairs (%d -> %d profiles)",
        n_pairs,
        len(profiles),
        len(fused),
    )
    return fused


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


def _add_bottom_cap_world(
    cloud: np.ndarray,
    *,
    grid_mm: float,
    bottom_quantile: float,
    border_pad_mm: float,
) -> np.ndarray:
    """Generate a single flat bottom cap in world frame (horizontal plane).

    Project convention: +Y is vertical (turntable axis), so the cap plane is
    constant Y. The support polygon is built in XZ.
    """
    xz = cloud[:, [0, 2]]
    if xz.shape[0] < 3:
        return np.empty((0, 3), dtype=np.float64)

    x0, x1 = float(np.quantile(xz[:, 0], 0.01)), float(np.quantile(xz[:, 0], 0.99))
    z0, z1 = float(np.quantile(xz[:, 1], 0.01)), float(np.quantile(xz[:, 1], 0.99))
    x0, x1 = x0 - border_pad_mm, x1 + border_pad_mm
    z0, z1 = z0 - border_pad_mm, z1 + border_pad_mm
    xs = np.arange(x0, x1 + grid_mm, grid_mm, dtype=np.float64)
    zs = np.arange(z0, z1 + grid_mm, grid_mm, dtype=np.float64)
    gx, gz = np.meshgrid(xs, zs, indexing="xy")

    hull = ConvexHull(xz)
    poly = xz[hull.vertices]
    mask = _points_in_polygon(gx.ravel(), gz.ravel(), poly).reshape(gx.shape)
    count = int(mask.sum())
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)

    y_bottom = float(np.quantile(cloud[:, 1], bottom_quantile))
    return np.column_stack(
        (gx[mask], np.full(count, y_bottom, dtype=np.float64), gz[mask])
    ).astype(np.float64)


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
    """Add a single flat bottom cap in world frame.

    This deliberately avoids top-cap generation and avoids PCA axis ambiguity
    by placing the cap on a horizontal plane (constant world Z).
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

    bot_w = _add_bottom_cap_world(
        cloud,
        grid_mm=grid_mm,
        bottom_quantile=bottom_quantile,
        border_pad_mm=border_pad_mm,
    )
    if bot_w.shape[0] == 0:
        logger.warning("add_flat_caps_aligned: no bottom cap points generated")
        return cloud

    merged = np.vstack((cloud, bot_w)).astype(np.float64)
    logger.info(
        "add_flat_caps_aligned: added bottom=%d points (world-horizontal cap)",
        bot_w.shape[0],
    )
    return merged
