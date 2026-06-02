"""scanner.reconstruction — Point cloud assembly and filtering.

Exports:
    merge_profiles: concatenate per-step 3D profiles into one cloud.
    filter_outliers: remove statistical outliers from a point cloud.
    fuse_half_turn_profiles: average duplicate profiles seen half a turn apart.
    clip_above_detected_top_plane: remove artifacts above a flat detected top.
"""

from scanner.reconstruction.pointcloud import (
    clip_above_detected_top_plane,
    filter_outliers,
    fuse_half_turn_profiles,
    merge_profiles,
)

from scanner.reconstruction.pointcloud import add_flat_caps_aligned

__all__ = [
    "merge_profiles",
    "filter_outliers",
    "add_flat_caps_aligned",
    "fuse_half_turn_profiles",
    "clip_above_detected_top_plane",
]
