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
    disable_background_filter,
    load_background_filter,
    save_background_filter,
)
from scanner.calibration.camera import (
    approximate_camera_intrinsics,
    calibrate_camera,
    load_camera_calibration,
)
from scanner.calibration.laser_plane import calibrate_laser_plane, load_laser_plane


class CalibrationError(Exception):
    """Raised when calibration data is missing, corrupt or mathematically invalid."""


__all__ = [
    "CalibrationError",
    "calibrate_camera",
    "approximate_camera_intrinsics",
    "save_background_filter",
    "load_background_filter",
    "disable_background_filter",
    "load_camera_calibration",
    "calibrate_laser_plane",
    "load_laser_plane",
]
