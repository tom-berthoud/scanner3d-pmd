"""Simple row-by-row laser line extraction.

The scanner uses a vertical laser line in the image.  For each image row, this
module keeps the pixels brighter than a threshold and returns one point at the
mean column of those active pixels.
"""

import logging
from collections.abc import Sequence
from typing import Any

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


def _edge_tolerance(size: int) -> int:
    """Return the near-edge snap tolerance in pixels for a camera axis."""
    return max(3, min(8, int(round(size * 0.01))))


def _snap_axis(value: float, low: int, high: int, tolerance: int) -> int:
    """Snap values close to an image border onto that border."""
    if value <= low + tolerance:
        return low
    if value >= high - tolerance:
        return high
    return int(round(value))


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


def _polygon_mask_sampled(
    shape: tuple[int, int],
    points: list[tuple[int, int]],
    x_offset: int,
    y_offset: int,
    x_stride: int,
    y_stride: int,
) -> np.ndarray:
    """Return a polygon mask on a sampled grid, using original image coordinates."""
    height, width = shape
    xs = np.asarray([point[0] for point in points], dtype=np.float64)
    ys = np.asarray([point[1] for point in points], dtype=np.float64)
    sample_x = (xs - float(x_offset)) / float(x_stride)
    sample_y = (ys - float(y_offset)) / float(y_stride)
    x0 = max(0, int(np.floor(sample_x.min())))
    x1 = min(width - 1, int(np.ceil(sample_x.max())))
    y0 = max(0, int(np.floor(sample_y.min())))
    y1 = min(height - 1, int(np.ceil(sample_y.max())))
    result = np.zeros(shape, dtype=bool)
    if x1 < x0 or y1 < y0:
        return result

    grid_x_idx, grid_y_idx = np.meshgrid(
        np.arange(x0, x1 + 1, dtype=np.float64),
        np.arange(y0, y1 + 1, dtype=np.float64),
    )
    grid_x = float(x_offset) + grid_x_idx * float(x_stride)
    grid_y = float(y_offset) + grid_y_idx * float(y_stride)

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
    original_shape: tuple[int, int] | None = None,
    x_offset: int = 0,
    y_offset: int = 0,
    x_stride: int = 1,
    y_stride: int = 1,
) -> np.ndarray:
    """Apply rectangular or polygonal exclusion masks to an active-pixel image."""
    if not shapes:
        return active

    masked = active.copy()
    height, width = masked.shape
    mask_height, mask_width = original_shape or masked.shape
    for shape in shapes:
        if not isinstance(shape, Sequence):
            continue

        if len(shape) == 4 and all(_is_number(value) for value in shape):
            x_tol = _edge_tolerance(mask_width)
            y_tol = _edge_tolerance(mask_height)
            x0 = _snap_axis(float(shape[0]), 0, mask_width, x_tol)
            y0 = _snap_axis(float(shape[1]), 0, mask_height, y_tol)
            x1 = _snap_axis(float(shape[2]), 0, mask_width, x_tol)
            y1 = _snap_axis(float(shape[3]), 0, mask_height, y_tol)
            x0 = max(0, min(mask_width, x0))
            x1 = max(0, min(mask_width, x1))
            y0 = max(0, min(mask_height, y0))
            y1 = max(0, min(mask_height, y1))
            if x1 > x0 and y1 > y0:
                col_coords = x_offset + np.arange(width) * x_stride
                row_coords = y_offset + np.arange(height) * y_stride
                cols = (col_coords >= x0) & (col_coords < x1)
                rows = (row_coords >= y0) & (row_coords < y1)
                masked[np.ix_(rows, cols)] = False
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
            x_tol = _edge_tolerance(mask_width)
            y_tol = _edge_tolerance(mask_height)
            points.append(
                (
                    max(0, min(mask_width, _snap_axis(float(x), 0, mask_width, x_tol))),
                    max(0, min(mask_height, _snap_axis(float(y), 0, mask_height, y_tol))),
                )
            )
        if len(points) < 3:
            logger.warning("Ignoring invalid laser mask shape: %s", shape)
            continue

        if original_shape is None and x_stride == 1 and y_stride == 1:
            masked[_polygon_mask(masked.shape, points)] = False
        else:
            masked[
                _polygon_mask_sampled(
                    masked.shape,
                    points,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    x_stride=x_stride,
                    y_stride=y_stride,
                )
            ] = False
    return masked


def extract_laser_line(
    frame: np.ndarray,
    threshold: int = 180,
    min_pixels: int = 10,
    subpixel: bool = True,
    mode: str = "row_mean",
    camera_id: str | None = None,
    mask_rects: Sequence[MaskShape] | None = None,
    x_stride: int = 1,
    y_stride: int = 1,
    x_offset: int = 0,
    y_offset: int = 0,
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
        x_stride: Horizontal sampling step. ``3`` inspects one column out of
            three while returning original image coordinates.
        y_stride: Vertical sampling step. ``2`` returns at most one point every
            two image rows.
        x_offset: First sampled column in original image coordinates.
        y_offset: First sampled row in original image coordinates.

    Returns:
        Float array of shape (N, 2), one ``[col, row]`` point per detected row.
    """
    _ = subpixel
    if mode != "row_mean":
        logger.warning("extract_laser_line: mode=%r ignored, only 'row_mean' is implemented", mode)

    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning("extract_laser_line: expected BGR frame (H,W,3), got %s", frame.shape)
        return _empty_line()

    height, width = frame.shape[:2]
    x_stride = max(1, int(x_stride))
    y_stride = max(1, int(y_stride))
    x_offset = max(0, min(width - 1, int(x_offset)))
    y_offset = max(0, min(height - 1, int(y_offset)))

    green = frame[y_offset::y_stride, x_offset::x_stride, 1]
    active = green >= int(threshold)

    rectangles: list[MaskShape] = []
    if camera_id:
        rectangles.extend(CAMERA_MASKS.get(str(camera_id), []))
    if mask_rects:
        rectangles.extend(mask_rects)
    active = _mask_shapes(
        active,
        rectangles,
        original_shape=(height, width),
        x_offset=x_offset,
        y_offset=y_offset,
        x_stride=x_stride,
        y_stride=y_stride,
    )

    points: list[tuple[float, float]] = []
    for row_idx in range(active.shape[0]):
        cols = np.flatnonzero(active[row_idx])
        if cols.size:
            col = float(x_offset + cols.astype(np.float64).mean() * x_stride)
            row = float(y_offset + row_idx * y_stride)
            points.append((col, row))

    if len(points) < max(int(min_pixels), 1):
        logger.debug(
            "extract_laser_line: only %d rows detected (min=%d, threshold=%d)",
            len(points),
            min_pixels,
            threshold,
        )
        return _empty_line()

    return np.asarray(points, dtype=np.float32)
