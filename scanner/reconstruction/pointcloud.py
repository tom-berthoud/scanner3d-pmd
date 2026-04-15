"""scanner.reconstruction.pointcloud — 3D point cloud operations.

Provides utilities to merge multiple scan profiles into a single point cloud
and to remove statistical outliers using a KD-tree based filter.
"""

import logging

import numpy as np

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
