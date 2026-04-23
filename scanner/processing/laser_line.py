"""scanner.processing.laser_line — Extract the laser line from a BGR frame.

The original implementation assumed a single vertical position per image
column.  That works for smooth horizontal profiles but collapses when the
laser line is vertical, strongly oblique, or locally broken by a concavity.

This version instead:
    1. isolates the green laser signal,
    2. thresholds and lightly closes the binary mask,
    3. keeps the strongest connected components,
    4. extracts a centreline along each component dominant axis,
    5. optionally interpolates only very short gaps.

The public API stays compatible: callers still receive a float array of
pixel coordinates ``[col, row]``.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CLOSE_KERNEL_SIZE = 3
_MAX_SHORT_GAP_BINS = 3
_COMPONENT_SCORE_RATIO = 0.18
_EXTRACTION_MODES = {"component_axis", "row_green"}


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
        logger.debug(
            "crop_laser_line: cutoff col=%.1f removed too many points (%d left, min=%d)",
            float(crop_left_of_col),
            filtered.shape[0],
            max(int(min_points), 1),
        )
        return _empty_line()

    order = np.lexsort((filtered[:, 1], filtered[:, 0]))
    return filtered[order].astype(np.float32, copy=False)


def _line_to_dense_rows(line: np.ndarray, image_height: int) -> np.ndarray:
    """Map a sparse ``[col, row]`` line to one column value per integer row."""
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError(f"line must be (N, 2), got {line.shape}")
    if image_height <= 0:
        raise ValueError(f"image_height must be > 0, got {image_height}")

    dense = np.full(int(image_height), np.nan, dtype=np.float64)
    if line.shape[0] == 0:
        return dense

    rows = np.rint(line[:, 1]).astype(np.int32)
    valid = (rows >= 0) & (rows < int(image_height))
    if not np.any(valid):
        return dense

    rows = rows[valid]
    cols = np.asarray(line[valid, 0], dtype=np.float64)

    sums = np.zeros(int(image_height), dtype=np.float64)
    counts = np.zeros(int(image_height), dtype=np.int32)
    np.add.at(sums, rows, cols)
    np.add.at(counts, rows, 1)

    row_mask = counts > 0
    dense[row_mask] = sums[row_mask] / counts[row_mask]
    return dense


def _dense_rows_to_line(dense_rows: np.ndarray) -> np.ndarray:
    """Convert a dense row-indexed column array back to sparse ``[col, row]``."""
    valid_rows = np.flatnonzero(~np.isnan(dense_rows))
    if valid_rows.size == 0:
        return _empty_line()

    cols = dense_rows[valid_rows].astype(np.float32)
    rows = valid_rows.astype(np.float32)
    return np.column_stack([cols, rows]).astype(np.float32, copy=False)


def _is_gap_fill_allowed(gap_size: int, max_gap: int | None) -> bool:
    if gap_size <= 0:
        return False
    if max_gap is None or int(max_gap) <= 0:
        return True
    return gap_size <= int(max_gap)


def _interpolate_circular_series(
    values: np.ndarray,
    max_gap: int | None = None,
) -> tuple[np.ndarray, int]:
    """Interpolate NaN gaps on a circular 1-D series."""
    series = np.asarray(values, dtype=np.float64).copy()
    valid = np.flatnonzero(~np.isnan(series))
    if valid.size < 2:
        return series, 0

    added = 0
    extended = np.concatenate([valid, valid[:1] + len(series)])
    for left, right in zip(extended[:-1], extended[1:]):
        gap = int(right - left - 1)
        if not _is_gap_fill_allowed(gap, max_gap):
            continue

        interp = np.linspace(
            series[left % len(series)],
            series[right % len(series)],
            gap + 2,
            dtype=np.float64,
        )[1:-1]
        for offset, value in enumerate(interp, start=1):
            idx = (left + offset) % len(series)
            if np.isnan(series[idx]):
                series[idx] = value
                added += 1

    return series, added


def fill_occluded_laser_gaps(
    line: np.ndarray,
    image_height: int,
    max_gap_rows: int | None = None,
    min_points: int = 1,
) -> np.ndarray:
    """Interpolate missing rows inside one laser profile.

    Only interior gaps are filled: rows before the first detection and after
    the last detection stay empty.
    """
    dense = _line_to_dense_rows(line, image_height=image_height)
    valid_rows = np.flatnonzero(~np.isnan(dense))
    if valid_rows.size < 2:
        return _dense_rows_to_line(dense)

    filled_rows = 0
    for left, right in zip(valid_rows[:-1], valid_rows[1:]):
        gap = int(right - left - 1)
        if not _is_gap_fill_allowed(gap, max_gap_rows):
            continue
        dense[left + 1 : right] = np.linspace(
            dense[left],
            dense[right],
            gap + 2,
            dtype=np.float64,
        )[1:-1]
        filled_rows += gap

    result = _dense_rows_to_line(dense)
    if result.shape[0] < max(int(min_points), 1):
        return _empty_line()

    if filled_rows > 0:
        logger.debug(
            "fill_occluded_laser_gaps: added %d row(s) inside one profile",
            filled_rows,
        )
    return result


def interpolate_laser_profiles(
    lines: list[np.ndarray],
    image_height: int,
    max_gap_frames: int | None = None,
    min_points: int = 1,
) -> list[np.ndarray]:
    """Fill occluded zones across successive scan views.

    For each image row, missing frames bracketed by two visible frames are
    linearly interpolated. This assumes the object is solid and that a hidden
    laser segment should be reconstructed between the last visible view and
    the next visible view where it reappears.
    """
    if not lines:
        return []

    dense_profiles = np.vstack(
        [_line_to_dense_rows(line, image_height=image_height) for line in lines]
    )
    filled = dense_profiles.copy()
    filled_points = 0

    for row_idx in range(filled.shape[1]):
        row_series, added = _interpolate_circular_series(
            dense_profiles[:, row_idx],
            max_gap=max_gap_frames,
        )
        filled[:, row_idx] = row_series
        filled_points += added

    # Enforce a solid profile on each frame: once the row extent of the object
    # is known, any remaining hole inside that extent is filled.
    top_rows = np.full(filled.shape[0], np.nan, dtype=np.float64)
    bottom_rows = np.full(filled.shape[0], np.nan, dtype=np.float64)
    for frame_idx in range(filled.shape[0]):
        valid_rows = np.flatnonzero(~np.isnan(filled[frame_idx]))
        if valid_rows.size == 0:
            continue
        top_rows[frame_idx] = float(valid_rows[0])
        bottom_rows[frame_idx] = float(valid_rows[-1])

    top_rows, top_added = _interpolate_circular_series(top_rows, max_gap=max_gap_frames)
    bottom_rows, bottom_added = _interpolate_circular_series(
        bottom_rows,
        max_gap=max_gap_frames,
    )
    filled_points += top_added + bottom_added

    for frame_idx in range(filled.shape[0]):
        if np.isnan(top_rows[frame_idx]) or np.isnan(bottom_rows[frame_idx]):
            continue

        top = max(0, int(round(top_rows[frame_idx])))
        bottom = min(image_height - 1, int(round(bottom_rows[frame_idx])))
        if bottom <= top:
            continue

        segment = filled[frame_idx, top : bottom + 1].copy()
        known = np.flatnonzero(~np.isnan(segment))
        if known.size == 0:
            continue
        if known.size == 1:
            segment[:] = segment[known[0]]
        else:
            segment[:] = np.interp(
                np.arange(segment.size, dtype=np.float64),
                known.astype(np.float64),
                segment[known],
            )
        filled_points += int(np.isnan(filled[frame_idx, top : bottom + 1]).sum())
        filled[frame_idx, top : bottom + 1] = segment

    results: list[np.ndarray] = []
    for dense in filled:
        line = _dense_rows_to_line(dense)
        if line.shape[0] < max(int(min_points), 1):
            results.append(_empty_line())
        else:
            results.append(line)

    if filled_points > 0:
        logger.debug(
            "interpolate_laser_profiles: added %d occluded sample(s) across frames",
            filled_points,
        )
    return results


def _compute_laser_signal(frame: np.ndarray) -> np.ndarray:
    """Build a green-dominant laser signal robust to white highlights."""
    blue = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    red = frame[:, :, 2].astype(np.int16)
    return np.clip(green - np.maximum(red, blue), 0, 255).astype(np.uint8)


def _extract_row_green_line(
    frame: np.ndarray,
    threshold: int,
    min_pixels: int,
    subpixel: bool,
) -> np.ndarray:
    """Extract one laser point per image row using the raw green channel."""
    green = frame[:, :, 1]
    points: list[tuple[float, float]] = []

    for row in range(green.shape[0]):
        cols = np.flatnonzero(green[row] >= threshold)
        if cols.size == 0:
            continue

        splits = np.where(np.diff(cols) > 1)[0] + 1
        segments = np.split(cols, splits)
        best_segment = max(
            segments,
            key=lambda seg: (int(green[row, seg].sum()), int(seg.size)),
        )

        if subpixel:
            weights = green[row, best_segment].astype(np.float64)
            weights = np.maximum(weights - float(threshold) + 1.0, 1.0)
            col_center = float(np.average(best_segment.astype(np.float64), weights=weights))
        else:
            col_center = float(
                int(round((int(best_segment[0]) + int(best_segment[-1])) / 2.0))
            )

        points.append((col_center, float(row)))

    if len(points) < min_pixels:
        logger.debug(
            "_extract_row_green_line: only %d rows detected (min=%d)",
            len(points),
            min_pixels,
        )
        return _empty_line()

    return np.asarray(points, dtype=np.float32)


def _principal_axis(points_xy: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the dominant direction of a weighted point cloud in image space."""
    centroid = np.average(points_xy, axis=0, weights=weights)
    centred = points_xy - centroid
    cov = (centred * weights[:, np.newaxis]).T @ centred / max(float(weights.sum()), 1.0)

    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]

    # Keep a stable direction for deterministic output ordering.
    if abs(axis[0]) >= abs(axis[1]):
        if axis[0] < 0.0:
            axis = -axis
    elif axis[1] < 0.0:
        axis = -axis

    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return axis / norm


def _extract_component_centerline(
    component_mask: np.ndarray,
    laser_signal: np.ndarray,
    subpixel: bool,
) -> np.ndarray:
    """Extract a centreline from one connected laser component."""
    rows, cols = np.nonzero(component_mask)
    if len(rows) == 0:
        return _empty_line()

    weights = laser_signal[rows, cols].astype(np.float64)
    weights = np.maximum(weights, 1.0)

    points_xy = np.column_stack([cols, rows]).astype(np.float64)
    axis = _principal_axis(points_xy, weights)

    centroid = np.average(points_xy, axis=0, weights=weights)
    rel = points_xy - centroid
    t_coords = rel @ axis

    t_min = float(np.floor(t_coords.min()))
    t_max = float(np.ceil(t_coords.max()))
    n_bins = max(1, int(t_max - t_min) + 1)
    bin_indices = np.clip(np.floor(t_coords - t_min).astype(np.int32), 0, n_bins - 1)

    binned_points = np.full((n_bins, 2), np.nan, dtype=np.float64)

    for bin_idx in np.unique(bin_indices):
        select = bin_indices == bin_idx
        bin_weights = weights[select]
        bin_points = points_xy[select]

        if subpixel:
            binned_points[bin_idx] = np.average(bin_points, axis=0, weights=bin_weights)
        else:
            center_idx = int(np.argmax(bin_weights))
            binned_points[bin_idx] = bin_points[center_idx]

    valid_bins = np.flatnonzero(~np.isnan(binned_points[:, 0]))
    for left, right in zip(valid_bins[:-1], valid_bins[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > _MAX_SHORT_GAP_BINS:
            continue
        start = binned_points[left]
        end = binned_points[right]
        for offset in range(1, gap + 1):
            alpha = offset / float(gap + 1)
            binned_points[left + offset] = (1.0 - alpha) * start + alpha * end

    result = binned_points[~np.isnan(binned_points[:, 0])]
    if result.shape[0] == 0:
        return _empty_line()

    return result.astype(np.float32)


def extract_laser_line(
    frame: np.ndarray,
    threshold: int = 180,
    min_pixels: int = 10,
    subpixel: bool = True,
    mode: str = "component_axis",
) -> np.ndarray:
    """Detect the green laser line in *frame* and return pixel coordinates.

    Args:
        frame: BGR image as numpy array of shape (H, W, 3), dtype uint8.
        threshold: Minimum laser signal to consider a pixel active.
        min_pixels: Minimum number of centreline points required to validate
            the full detection.
        subpixel: If True, use weighted centroids inside local bins.  If False,
            keep the strongest pixel per bin.
        mode: Extraction strategy.
            ``component_axis``: connected components + dominant axis centreline.
            ``row_green``: one point per image row from the raw green channel.

    Returns:
        Float array of shape (N, 2) where each row is ``[col, row]``.
        Returns an empty array if no plausible laser line is detected.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        logger.warning(
            "extract_laser_line: expected BGR frame (H,W,3), got shape %s", frame.shape
        )
        return _empty_line()

    if mode not in _EXTRACTION_MODES:
        raise ValueError(
            f"Unknown extraction mode {mode!r}; expected one of {sorted(_EXTRACTION_MODES)}"
        )

    if mode == "row_green":
        return _extract_row_green_line(
            frame,
            threshold=threshold,
            min_pixels=min_pixels,
            subpixel=subpixel,
        )

    laser_signal = _compute_laser_signal(frame)
    mask = laser_signal >= threshold  # type: ignore[operator]
    if not mask.any():
        logger.debug("extract_laser_line: no pixels above threshold=%d", threshold)
        return _empty_line()

    import cv2  # type: ignore[import]

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((_CLOSE_KERNEL_SIZE, _CLOSE_KERNEL_SIZE), dtype=np.uint8)
    mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)

    components: list[tuple[float, np.ndarray]] = []
    min_component_area = max(4, min_pixels // 3)
    min_component_points = max(3, min_pixels // 3)

    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        component_mask = labels == label_idx
        centerline = _extract_component_centerline(component_mask, laser_signal, subpixel=subpixel)
        if centerline.shape[0] < min_component_points:
            continue

        score = float(laser_signal[component_mask].sum())
        components.append((score, centerline))

    if not components:
        logger.debug("extract_laser_line: no connected component survived filtering")
        return _empty_line()

    components.sort(key=lambda item: item[0], reverse=True)
    best_score = components[0][0]

    kept_segments = [
        segment
        for score, segment in components
        if score >= best_score * _COMPONENT_SCORE_RATIO or segment.shape[0] >= min_pixels
    ]

    result = np.vstack(kept_segments).astype(np.float32)

    # Stable ordering makes debugging and tests easier.
    order = np.lexsort((result[:, 1], result[:, 0]))
    result = result[order]

    if result.shape[0] < min_pixels:
        logger.debug(
            "extract_laser_line: only %d points after filtering (min=%d)",
            result.shape[0],
            min_pixels,
        )
        return _empty_line()

    logger.debug(
        "extract_laser_line: detected %d points across %d segment(s)",
        len(result),
        len(kept_segments),
    )
    return result
