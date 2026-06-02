"""scanner.reconstruction — Point cloud assembly and filtering.

Exports:
    merge_profiles: concatenate per-step 3D profiles into one cloud.
    filter_outliers: remove statistical outliers from a point cloud.
    fuse_half_turn_profiles: average duplicate profiles seen half a turn apart.
"""

from scanner.reconstruction.pointcloud import filter_outliers, fuse_half_turn_profiles, merge_profiles

from scanner.reconstruction.pointcloud import add_flat_caps_aligned

__all__ = ["merge_profiles", "filter_outliers", "add_flat_caps_aligned", "fuse_half_turn_profiles"]
