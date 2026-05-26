"""Estimate alignment offset between two point clouds already in global frame.

Typical usage:
    python test_debug/estimate_cloud_offset.py \
      --source /tmp/scans/scan_cloud_left.ply \
      --target /tmp/scans/scan_cloud_right.ply

Outputs:
    - best rigid transform (R, t) mapping source -> target
    - translation-only offset estimate
    - RMS nearest-neighbor error before/after
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        pts = np.load(path)
        return _as_xyz(pts, path)
    if suffix in {".csv", ".txt"}:
        pts = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        return _as_xyz(pts, path)
    if suffix in {".ply", ".pcd"}:
        try:
            import open3d as o3d
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "open3d is required for .ply/.pcd inputs. Install requirements first."
            ) from exc
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
        return _as_xyz(pts, path)
    raise ValueError(f"Unsupported format: {path}")


def _as_xyz(points: np.ndarray, path: Path) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path}: expected Nx3 array, got shape {arr.shape}")
    arr = arr[:, :3]
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    if arr.shape[0] == 0:
        raise ValueError(f"{path}: no valid finite 3D points")
    return arr


def _downsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    return r, t


def _rms_nn(a: np.ndarray, b: np.ndarray) -> float:
    tree = cKDTree(b)
    dists, _ = tree.query(a, k=1)
    return float(np.sqrt(np.mean(dists**2)))


def estimate_transform_icp(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int = 25,
    trim_percent: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point ICP with nearest-neighbor correspondences.

    Returns:
        (R, t, rms_after) where target ~= (R @ source.T).T + t.
    """
    r_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    moved = source.copy()
    tree = cKDTree(target)

    for _ in range(iterations):
        dists, idx = tree.query(moved, k=1)
        matched = target[idx]

        # Trim worst pairs for robustness.
        keep_n = max(10, int(trim_percent * moved.shape[0]))
        keep_idx = np.argsort(dists)[:keep_n]
        src_k = moved[keep_idx]
        dst_k = matched[keep_idx]

        r_step, t_step = _kabsch(src_k, dst_k)
        moved = (r_step @ moved.T).T + t_step

        r_total = r_step @ r_total
        t_total = r_step @ t_total + t_step

    rms_after = _rms_nn(moved, target)
    return r_total, t_total, rms_after


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Source cloud to move.")
    parser.add_argument("--target", required=True, type=Path, help="Target reference cloud.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Randomly keep up to this many points per cloud.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--iterations", type=int, default=30, help="ICP iterations.")
    parser.add_argument(
        "--trim-percent",
        type=float,
        default=0.85,
        help="Fraction of closest pairs used each ICP iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _downsample(_load_points(args.source), max_points=args.max_points, seed=args.seed)
    target = _downsample(_load_points(args.target), max_points=args.max_points, seed=args.seed + 1)

    rms_before = _rms_nn(source, target)
    r, t, rms_after = estimate_transform_icp(
        source,
        target,
        iterations=args.iterations,
        trim_percent=args.trim_percent,
    )

    # Translation-only estimate from centroids for quick extrinsics tweak.
    t_centroid = target.mean(axis=0) - source.mean(axis=0)

    result = {
        "source_points": int(source.shape[0]),
        "target_points": int(target.shape[0]),
        "rms_before_mm": rms_before,
        "rms_after_mm": rms_after,
        "translation_only_offset_mm": t_centroid.tolist(),
        "rigid_transform": {
            "rotation_matrix": r.tolist(),
            "translation_mm": t.tolist(),
        },
    }
    print(json.dumps(result, indent=2))

    print("\nSuggested quick fix:")
    print(
        "  Start with translation_only_offset_mm on the drifting camera extrinsics,"
        " then use rigid_transform.translation_mm if needed."
    )


if __name__ == "__main__":
    main()

