"""scanner.processing — Laser line extraction and 3D triangulation.

Exports:
    extract_laser_line: detect the green laser line in a BGR frame.
    triangulate: convert line pixels to 3D world coordinates.
"""

from scanner.processing.laser_line import extract_laser_line
from scanner.processing.triangulation import triangulate

__all__ = ["extract_laser_line", "triangulate"]
