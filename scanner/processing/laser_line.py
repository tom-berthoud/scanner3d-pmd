"""scanner.processing.laser_line — Extract the laser line from a BGR frame.

Uses the spectral advantage of the 520 nm green laser:
    laser_signal = clip(G - R, 0, 255)
    (see agents.md §2 — G/R contrast ratio ≈ 30:1 at 520 nm)

After thresholding, the sub-pixel centroid of each column is computed to
give sub-pixel accuracy on the vertical laser line position.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_laser_line(
    frame: np.ndarray,
    threshold: int = 180,
    min_pixels: int = 10,
    subpixel: bool = True,
) -> np.ndarray:
    """Detect the green laser line in *frame* and return pixel coordinates.

    Algorithm:
        1. Compute laser_signal = clip(G.astype(int) - R.astype(int), 0, 255)
        2. Apply binary threshold at *threshold*
        3. For each column with at least one pixel above threshold, compute
           the intensity-weighted centroid (sub-pixel row) or the argmax row.
        4. Return only columns that contribute to a continuous line of at
           least *min_pixels* columns.

    Args:
        frame: BGR image as numpy array of shape (H, W, 3), dtype uint8.
        threshold: Minimum laser_signal value to consider a pixel active.
            Range 0–255.  Default 180 matches the laser threshold in
            settings.yaml.
        min_pixels: Minimum number of active columns required to consider a
            line detected.  Below this the function returns an empty array.
        subpixel: If True, use intensity-weighted centroid for sub-pixel
            accuracy.  If False, use integer argmax.

    Returns:
        Float array of shape (N, 2) where each row is [col, row] in pixel
        coordinates.  Returns an empty array of shape (0, 2) if the laser
        line is not detected.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning(
            "extract_laser_line: expected BGR frame (H,W,3), got shape %s", frame.shape
        )
        return np.empty((0, 2), dtype=np.float32)

    # BGR → separate channels
    blue = frame[:, :, 0].astype(np.int32)
    green = frame[:, :, 1].astype(np.int32)
    red = frame[:, :, 2].astype(np.int32)

    # Suppress ambient light using colour difference (agents.md §2)
    laser_signal = np.clip(green - red, 0, 255).astype(np.uint8)

    # Apply threshold → binary mask (H, W)
    mask = laser_signal >= threshold  # type: ignore[operator]

    height, width = frame.shape[:2]
    row_indices = np.arange(height, dtype=np.float64)

    cols: list[float] = []
    rows: list[float] = []

    for col in range(width):
        col_mask = mask[:, col]
        if not col_mask.any():
            continue

        col_signal = laser_signal[:, col].astype(np.float64)
        col_signal_masked = col_signal * col_mask

        total = col_signal_masked.sum()
        if total < 1e-9:
            continue

        if subpixel:
            # Intensity-weighted centroid — sub-pixel accuracy
            row_f = (row_indices * col_signal_masked).sum() / total
        else:
            row_f = float(np.argmax(col_signal_masked))

        cols.append(float(col))
        rows.append(row_f)

    if len(cols) < min_pixels:
        logger.debug(
            "extract_laser_line: only %d active columns (min=%d) — no line detected",
            len(cols),
            min_pixels,
        )
        return np.empty((0, 2), dtype=np.float32)

    result = np.column_stack([cols, rows]).astype(np.float32)
    logger.debug(
        "extract_laser_line: detected %d line pixels (subpixel=%s)", len(result), subpixel
    )
    return result
