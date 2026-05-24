"""scanner.calibration — Camera and laser plane calibration utilities.

Exports:
    CalibrationError: raised when calibration data is missing or invalid.
    calibrate_camera: perform intrinsic camera calibration from checkerboard images.
    disable_background_filter: disable the saved left-image crop.
    load_background_filter: load the saved left-image crop.
    load_camera_calibration: load camera matrix and dist_coeffs from YAML.
    calibrate_laser_plane: fit the laser plane from reference measurements.
    load_laser_plane: load the laser plane equation from YAML.
    save_background_filter: persist the saved left-image crop.
"""

from scanner.calibration.background_filter import (
    background_crop_left_col,
    disable_background_filter,
    load_background_filter,
    save_background_filter,
)
from scanner.calibration.camera import (
    approximate_camera_intrinsics,
    calibrate_camera,
    calibrate_camera_with_report,
    checkerboard_capture_quality,
    draw_checkerboard_overlay,
    load_camera_calibration,
)
from scanner.calibration.laser_plane import calibrate_laser_plane, load_laser_plane
from scanner.calibration.multi_camera import (
    camera_config_by_id,
    camera_configs,
    camera_ids,
    default_camera_id,
    load_camera_model,
)


class CalibrationError(Exception):
    """Raised when calibration data is missing, corrupt or mathematically invalid."""


__all__ = [
    "CalibrationError",
    "calibrate_camera",
    "calibrate_camera_with_report",
    "checkerboard_capture_quality",
    "draw_checkerboard_overlay",
    "approximate_camera_intrinsics",
    "background_crop_left_col",
    "save_background_filter",
    "load_background_filter",
    "disable_background_filter",
    "load_camera_calibration",
    "calibrate_laser_plane",
    "load_laser_plane",
    "camera_configs",
    "camera_config_by_id",
    "camera_ids",
    "default_camera_id",
    "load_camera_model",
]
