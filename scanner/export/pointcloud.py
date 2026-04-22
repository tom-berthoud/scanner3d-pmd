"""scanner.export.pointcloud — Export raw point clouds for inspection.

The mesh export can hide reconstruction issues by bridging gaps or smoothing
concavities.  Saving the raw cloud alongside the STL/OBJ makes debugging much
easier.
"""

import os

import numpy as np


def export_point_cloud_ply(cloud: np.ndarray, path: str) -> None:
    """Export *cloud* as an ASCII PLY point cloud.

    Args:
        cloud: Float array of shape (N, 3).
        path: Destination file path, typically ending with ``.ply``.

    Raises:
        ValueError: if *cloud* is not a non-empty ``(N, 3)`` array.
    """
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {cloud.shape}")
    if cloud.shape[0] == 0:
        raise ValueError("Need at least 1 point to export a point cloud")

    cloud_f64 = np.asarray(cloud, dtype=np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "comment scanner3d-pmd raw point cloud",
            f"element vertex {cloud_f64.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
    )

    with open(path, "w", encoding="ascii") as fh:
        fh.write(header)
        fh.write("\n")
        for x, y, z in cloud_f64:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
