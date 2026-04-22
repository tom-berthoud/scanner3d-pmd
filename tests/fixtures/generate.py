"""tests.fixtures.generate — Synthetic image generator for unit tests.

Creates BGR images with a simulated green laser line at a configurable
vertical position.  Used by test_processing.py and related tests.
"""

import math

import numpy as np


def make_laser_frame(
    width: int = 640,
    height: int = 480,
    row: float = 240.0,
    laser_width_px: float = 3.0,
    laser_intensity: int = 220,
    ambient_green: int = 5,
    ambient_other: int = 2,
    noise_amplitude: int = 3,
    rng_seed: int = 42,
) -> np.ndarray:
    """Create a synthetic BGR image with a horizontal laser line.

    The laser line is bright green (G ≈ laser_intensity, R ≈ B ≈ low),
    which is the pattern produced by a 520 nm green laser with the IMX708
    sensor (see agents.md §2).

    The line is drawn across the full width using a Gaussian cross-section
    for sub-pixel testing.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        row: Vertical position of the laser line centre (can be fractional).
        laser_width_px: 1-sigma of the Gaussian cross-section.
        laser_intensity: Peak green channel value (0–255).
        ambient_green: Background green level.
        ambient_other: Background R and B level.
        noise_amplitude: Random noise amplitude added to all channels.
        rng_seed: Random seed for reproducibility.

    Returns:
        BGR image as numpy array of shape (height, width, 3), dtype uint8.
    """
    rng = np.random.default_rng(rng_seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Ambient background
    frame[:, :, 0] = ambient_other  # B
    frame[:, :, 1] = ambient_green  # G
    frame[:, :, 2] = ambient_other  # R

    # Draw laser line with Gaussian cross-section
    rows_f = np.arange(height, dtype=np.float64)
    gaussian = np.exp(-0.5 * ((rows_f - row) / laser_width_px) ** 2)
    green_channel = (laser_intensity * gaussian).clip(0, 255).astype(np.uint8)

    for col in range(width):
        frame[:, col, 1] = np.maximum(frame[:, col, 1], green_channel)

    # Add noise
    if noise_amplitude > 0:
        noise = rng.integers(0, noise_amplitude + 1, (height, width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def make_laser_frame_sphere(
    width: int = 640,
    height: int = 480,
    angle_rad: float = 0.0,
    sphere_radius_px: int = 120,
    cx: int = 320,
    cy: int = 240,
    laser_intensity: int = 220,
    rng_seed: int = 42,
) -> np.ndarray:
    """Create a synthetic BGR image with a sphere-profile laser line.

    The laser line follows the visible surface of a sphere seen from the
    front.  The angular parameter varies the visible stripe position, giving
    the appearance of a rotating turntable.

    Args:
        width: Image width.
        height: Image height.
        angle_rad: Turntable rotation angle in radians.
        sphere_radius_px: Sphere radius in pixels.
        cx: Horizontal centre of the sphere in pixels.
        cy: Vertical centre of the sphere in pixels.
        laser_intensity: Peak green channel value.
        rng_seed: Random seed.

    Returns:
        BGR image as numpy array of shape (height, width, 3), dtype uint8.
    """
    rng = np.random.default_rng(rng_seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Faint background noise
    noise = rng.integers(0, 4, (height, width, 3), dtype=np.uint8)
    frame += noise

    triang_angle = math.radians(30.0)
    y_shift = sphere_radius_px * math.cos(angle_rad) * math.sin(triang_angle)
    slice_radius = sphere_radius_px * abs(math.sin(angle_rad)) if abs(math.sin(angle_rad)) > 0.05 else sphere_radius_px * 0.05

    for col in range(width):
        dx = (col - cx) / max(slice_radius, 1.0)
        if abs(dx) > 1.0:
            continue
        arc_y = slice_radius * math.sqrt(max(0.0, 1.0 - dx * dx))
        row_f = cy - y_shift - arc_y
        row = int(round(row_f))
        for dr in range(-3, 4):
            r = row + dr
            if 0 <= r < height:
                intensity = int(laser_intensity * math.exp(-0.5 * (dr / 1.5) ** 2))
                frame[r, col, 1] = min(255, intensity)

    return frame


def make_laser_frame_vertical(
    width: int = 640,
    height: int = 480,
    col: float = 320.0,
    laser_width_px: float = 3.0,
    laser_intensity: int = 220,
    ambient_green: int = 5,
    ambient_other: int = 2,
    noise_amplitude: int = 3,
    rng_seed: int = 42,
) -> np.ndarray:
    """Create a synthetic BGR image with a vertical laser line."""
    rng = np.random.default_rng(rng_seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    frame[:, :, 0] = ambient_other
    frame[:, :, 1] = ambient_green
    frame[:, :, 2] = ambient_other

    cols_f = np.arange(width, dtype=np.float64)
    gaussian = np.exp(-0.5 * ((cols_f - col) / laser_width_px) ** 2)
    green_profile = (laser_intensity * gaussian).clip(0, 255).astype(np.uint8)

    for row in range(height):
        frame[row, :, 1] = np.maximum(frame[row, :, 1], green_profile)

    if noise_amplitude > 0:
        noise = rng.integers(0, noise_amplitude + 1, (height, width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def make_laser_frame_polyline(
    points: list[tuple[float, float]],
    width: int = 640,
    height: int = 480,
    laser_width_px: float = 1.6,
    laser_intensity: int = 220,
    ambient_green: int = 5,
    ambient_other: int = 2,
    noise_amplitude: int = 0,
    rng_seed: int = 42,
    gap_segments: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Create a frame with a laser line following a polyline in image space.

    Args:
        points: Ordered list of ``(col, row)`` control points.
        gap_segments: Optional list of ``(start_ratio, end_ratio)`` intervals
            skipped along the polyline, where 0 is the first point and 1 the
            last point.
    """
    rng = np.random.default_rng(rng_seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = ambient_other
    frame[:, :, 1] = ambient_green
    frame[:, :, 2] = ambient_other

    if len(points) < 2:
        return frame

    skip_ranges = gap_segments or []

    def _is_skipped(progress: float) -> bool:
        return any(start <= progress <= end for start, end in skip_ranges)

    segment_lengths: list[float] = []
    total_length = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seg_len = float(math.hypot(x1 - x0, y1 - y0))
        segment_lengths.append(seg_len)
        total_length += seg_len

    traversed = 0.0
    for (x0, y0), (x1, y1), seg_len in zip(points[:-1], points[1:], segment_lengths):
        steps = max(2, int(math.ceil(seg_len * 2.5)))
        for step in range(steps + 1):
            alpha = step / float(steps)
            progress = (traversed + alpha * seg_len) / max(total_length, 1e-9)
            if _is_skipped(progress):
                continue

            col_f = x0 + alpha * (x1 - x0)
            row_f = y0 + alpha * (y1 - y0)
            col_i = int(round(col_f))
            row_i = int(round(row_f))

            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    rr = row_i + dy
                    cc = col_i + dx
                    if 0 <= rr < height and 0 <= cc < width:
                        dist2 = dx * dx + dy * dy
                        intensity = int(laser_intensity * math.exp(-0.5 * dist2 / max(laser_width_px**2, 1e-6)))
                        frame[rr, cc, 1] = max(frame[rr, cc, 1], intensity)
        traversed += seg_len

    if noise_amplitude > 0:
        noise = rng.integers(0, noise_amplitude + 1, (height, width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame
