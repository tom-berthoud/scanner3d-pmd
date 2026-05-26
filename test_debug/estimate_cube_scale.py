"""Estimate per-axis and isotropic scale between two cube point clouds.

This script assumes both clouds are already in the same global frame and come
from the same rigid cube scan (one cloud per camera).

Example:
    python test_debug/estimate_cube_scale.py \
      --source /tmp/scans/scan_..._cloud_left.ply \
      --target /tmp/scans/scan_..._cloud_right.ply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        pts = np.load(path)
    elif suffix in {".csv", ".txt"}:
        pts = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix in {".ply", ".pcd"}:
        try:
            import open3d as o3d
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("open3d is required for .ply/.pcd inputs") from exc
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported format: {path}")

    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"{path}: expected Nx3, got {pts.shape}")
    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        raise ValueError(f"{path}: no valid points")
    return pts


def _downsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _robust_extent(points: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    low = np.percentile(points, q_low, axis=0)
    high = np.percentile(points, q_high, axis=0)
    extent = high - low
    return np.maximum(extent, 1e-9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Cloud to scale (source camera).")
    parser.add_argument("--target", required=True, type=Path, help="Reference cloud (target camera).")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-low", type=float, default=5.0, help="Low percentile for robust extents.")
    parser.add_argument("--q-high", type=float, default=95.0, help="High percentile for robust extents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = _downsample(_load_points(args.source), args.max_points, args.seed)
    tgt = _downsample(_load_points(args.target), args.max_points, args.seed + 1)

    # Remove translation influence before extent comparison.
    src_c = src - src.mean(axis=0)
    tgt_c = tgt - tgt.mean(axis=0)

    src_extent = _robust_extent(src_c, args.q_low, args.q_high)
    tgt_extent = _robust_extent(tgt_c, args.q_low, args.q_high)

    scale_xyz = tgt_extent / src_extent
    scale_iso = float(np.mean(scale_xyz))

    out = {
        "source_points": int(src.shape[0]),
        "target_points": int(tgt.shape[0]),
        "percentiles": {"low": args.q_low, "high": args.q_high},
        "source_extent_mm": src_extent.tolist(),
        "target_extent_mm": tgt_extent.tolist(),
        "scale_xyz_source_to_target": scale_xyz.tolist(),
        "scale_isotropic_source_to_target": scale_iso,
        "yaml_snippet_for_source_camera": {
            "extrinsics": {
                "scale_xyz": [float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])]
            }
        },
    }
    print(json.dumps(out, indent=2))

    print("\nSuggested usage:")
    print("  Put scale_xyz into the SOURCE camera extrinsics in config/settings.yaml")
    print("  Re-scan and iterate until both camera clouds overlap.")


if __name__ == "__main__":
    main()

