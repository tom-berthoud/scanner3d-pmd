"""Simple row-by-row laser line extraction.

The scanner uses a vertical laser line in the image.  For each image row, this
module keeps the pixels brighter than a threshold and returns one point at the
mean column of those active pixels.
"""

import logging
from typing import Any
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

MaskShape = Sequence[Any]

# Masks are deliberately empty for now. Each item can be either:
# - rectangle: [x0, y0, x1, y1] with x1/y1 excluded
# - polygon/trapezoid: [[x0, y0], [x1, y1], [x2, y2], ...]
CAMERA_MASKS: dict[str, list[MaskShape]] = {
    "right": [],  # Nappe / Raspberry Pi Camera 3
    "left": [],  # USB Arducam
    "nappe": [],
    "usb": [],
}


def _empty_line() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    """Return a boolean mask for the polygon defined by *points*."""
    height, width = shape
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    x0 = max(0, int(np.floor(xs.min())))
    x1 = min(width - 1, int(np.ceil(xs.max())))
    y0 = max(0, int(np.floor(ys.min())))
    y1 = min(height - 1, int(np.ceil(ys.max())))
    result = np.zeros(shape, dtype=bool)
    if x1 < x0 or y1 < y0:
        return result

    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.float64) + 0.5,
        np.arange(y0, y1 + 1, dtype=np.float64) + 0.5,
    )
    inside = np.zeros(grid_x.shape, dtype=bool)
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        crosses = (yi > grid_y) != (yj > grid_y)
        x_intersection = (xj - xi) * (grid_y - yi) / ((yj - yi) or 1e-12) + xi
        inside ^= crosses & (grid_x < x_intersection)
        j = i

    result[y0 : y1 + 1, x0 : x1 + 1] = inside
    return result


def _mask_shapes(
    active: np.ndarray,
    shapes: Sequence[MaskShape] | None,
) -> np.ndarray:
    """Apply rectangular or polygonal exclusion masks to an active-pixel image."""
    if not shapes:
        return active

    masked = active.copy()
    height, width = masked.shape
    for shape in shapes:
        if not isinstance(shape, Sequence):
            continue

        if len(shape) == 4 and all(_is_number(value) for value in shape):
            x0, y0, x1, y1 = [int(value) for value in shape]
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            if x1 > x0 and y1 > y0:
                masked[y0:y1, x0:x1] = False
            continue

        points: list[tuple[int, int]] = []
        for point in shape:
            if not isinstance(point, Sequence) or len(point) != 2:
                points = []
                break
            x, y = point
            if not _is_number(x) or not _is_number(y):
                points = []
                break
            points.append(
                (
                    max(0, min(width - 1, int(round(float(x))))),
                    max(0, min(height - 1, int(round(float(y))))),
                )
            )
        if len(points) < 3:
            logger.warning("Ignoring invalid laser mask shape: %s", shape)
            continue

        masked[_polygon_mask(masked.shape, points)] = False
    return masked


def extract_laser_line(
    frame: np.ndarray,
    threshold: int = 180,
    min_pixels: int = 10,
    subpixel: bool = True,
    mode: str = "row_mean",
    camera_id: str | None = None,
    mask_rects: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Extract one laser point per image row.

    Args:
        frame: BGR image as numpy array of shape (H, W, 3), dtype uint8.
        threshold: Minimum green-channel value for a pixel to count as lit.
        min_pixels: Minimum number of detected rows required for a valid line.
        subpixel: Kept for API compatibility. The row mean is always float.
        mode: Kept for API compatibility. All modes use this simple extractor.
        camera_id: Optional camera id used to select a predefined mask.
        mask_rects: Optional exclusion masks. Each item can be a rectangle
            ``[x0, y0, x1, y1]`` or a polygon ``[[x0, y0], ...]``.

    Returns:
        Float array of shape (N, 2), one ``[col, row]`` point per detected row.
    """
    del subpixel, mode

    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning("extract_laser_line: expected BGR frame (H,W,3), got %s", frame.shape)
        return _empty_line()

    green = frame[:, :, 1]
    active = green >= int(threshold)

    rectangles: list[MaskShape] = []
    if camera_id:
        rectangles.extend(CAMERA_MASKS.get(str(camera_id), []))
    if mask_rects:
        rectangles.extend(mask_rects)
    active = _mask_shapes(active, rectangles)

    points: list[tuple[float, float]] = []
    for row_idx in range(active.shape[0]):
        cols = np.flatnonzero(active[row_idx])
        if cols.size:
            points.append((float(cols.mean()), float(row_idx)))

    if len(points) < max(int(min_pixels), 1):
        logger.debug(
            "extract_laser_line: only %d rows detected (min=%d, threshold=%d)",
            len(points),
            min_pixels,
            threshold,
        )
        return _empty_line()

    return np.asarray(points, dtype=np.float32)
