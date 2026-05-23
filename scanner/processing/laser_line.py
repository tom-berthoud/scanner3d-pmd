"""Simple row-by-row laser line extraction.

The scanner uses a vertical laser line in the image.  For each image row, this
module keeps the pixels brighter than a threshold and returns one point at the
mean column of those active pixels.
"""

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)

MaskRect = tuple[int, int, int, int]

# Masks are deliberately empty for now.  Each rectangle is (x0, y0, x1, y1)
# with x1/y1 excluded, in image pixel coordinates.
CAMERA_MASKS: dict[str, list[MaskRect]] = {
    "right": [],  # Nappe / Raspberry Pi Camera 3
    "left": [],  # USB Arducam
    "nappe": [],
    "usb": [],
}


def _empty_line() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def crop_laser_line(
    line: np.ndarray,
    crop_left_of_col: float | None = None,
    min_points: int = 1,
) -> np.ndarray:
    """Remove detected points left of a calibrated image-column cutoff."""
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError(f"line must be (N, 2), got {line.shape}")

    if line.shape[0] == 0 or crop_left_of_col is None:
        return line.astype(np.float32, copy=False)

    filtered = np.asarray(line, dtype=np.float32)
    filtered = filtered[filtered[:, 0] >= float(crop_left_of_col)]
    if filtered.shape[0] < max(int(min_points), 1):
        return _empty_line()

    order = np.lexsort((filtered[:, 0], filtered[:, 1]))
    return filtered[order].astype(np.float32, copy=False)


def _mask_rectangles(
    active: np.ndarray,
    rectangles: Sequence[Sequence[int]] | None,
) -> np.ndarray:
    """Apply rectangular exclusion masks to an active-pixel image."""
    if not rectangles:
        return active

    masked = active.copy()
    height, width = masked.shape
    for rect in rectangles:
        if len(rect) != 4:
            logger.warning("Ignoring invalid laser mask rectangle: %s", rect)
            continue
        x0, y0, x1, y1 = [int(value) for value in rect]
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 > x0 and y1 > y0:
            masked[y0:y1, x0:x1] = False
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
        mask_rects: Optional rectangular masks, each as ``[x0, y0, x1, y1]``.

    Returns:
        Float array of shape (N, 2), one ``[col, row]`` point per detected row.
    """
    del subpixel, mode

    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning("extract_laser_line: expected BGR frame (H,W,3), got %s", frame.shape)
        return _empty_line()

    green = frame[:, :, 1]
    active = green >= int(threshold)

    rectangles: list[Sequence[int]] = []
    if camera_id:
        rectangles.extend(CAMERA_MASKS.get(str(camera_id), []))
    if mask_rects:
        rectangles.extend(mask_rects)
    active = _mask_rectangles(active, rectangles)

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
