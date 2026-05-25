"""Generate a printable A4 PDF with five 20 mm ArUco markers.

The marker data is copied from OpenCV's predefined DICT_4X4_1000_BYTES.
DICT_4X4_50 uses the first 50 entries of that table; this sheet uses IDs 0..4.
Source:
https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/predefined_dictionaries.hpp
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MM_PER_INCH = 25.4
A4_W_MM = 210.0
A4_H_MM = 297.0
MARKER_SIZE_MM = 20.0
BORDER_BITS = 1
INNER_BITS = 4

# OpenCV DICT_4X4_50, IDs 0..4, normal rotation bytes only.
ARUCO_4X4_50_BYTES: dict[int, tuple[int, int]] = {
    0: (181, 50),
    1: (15, 154),
    2: (51, 45),
    3: (153, 70),
    4: (84, 158),
}


def bits_from_bytes(byte_pair: tuple[int, int]) -> np.ndarray:
    """Return the 4x4 ArUco payload bits in row-major order."""
    values = []
    for byte in byte_pair:
        for shift in range(7, -1, -1):
            values.append((byte >> shift) & 1)
    return np.array(values, dtype=np.uint8).reshape(INNER_BITS, INNER_BITS)


def marker_image(marker_id: int) -> np.ndarray:
    """Return a 6x6 marker image with one black border bit."""
    size = INNER_BITS + 2 * BORDER_BITS
    image = np.zeros((size, size), dtype=np.uint8)
    image[BORDER_BITS:-BORDER_BITS, BORDER_BITS:-BORDER_BITS] = bits_from_bytes(
        ARUCO_4X4_50_BYTES[marker_id]
    )
    return image


def add_marker(fig: plt.Figure, marker_id: int, left_mm: float, bottom_mm: float) -> None:
    """Place one marker with exact physical dimensions on the page."""
    ax = fig.add_axes(
        [
            left_mm / A4_W_MM,
            bottom_mm / A4_H_MM,
            MARKER_SIZE_MM / A4_W_MM,
            MARKER_SIZE_MM / A4_H_MM,
        ]
    )
    ax.imshow(marker_image(marker_id), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()

    fig.text(
        (left_mm + MARKER_SIZE_MM / 2) / A4_W_MM,
        (bottom_mm - 5) / A4_H_MM,
        f"DICT_4X4_50 ID {marker_id} - 20 mm",
        ha="center",
        va="top",
        fontsize=7,
    )


def main() -> None:
    out_path = Path(__file__).with_name("aruco_4x4_20mm_A4.pdf")
    fig = plt.figure(figsize=(A4_W_MM / MM_PER_INCH, A4_H_MM / MM_PER_INCH), dpi=300)
    fig.patch.set_facecolor("white")

    positions = [
        (0, 45.0, 215.0),
        (1, 145.0, 215.0),
        (2, 45.0, 145.0),
        (3, 145.0, 145.0),
        (4, 95.0, 105.0),
    ]
    for marker_id, left_mm, bottom_mm in positions:
        add_marker(fig, marker_id, left_mm, bottom_mm)

    # Physical scale check: this line must measure 50 mm after printing at 100%.
    ax = fig.add_axes([45.0 / A4_W_MM, 75.0 / A4_H_MM, 50.0 / A4_W_MM, 0.001])
    ax.plot([0, 1], [0, 0], color="black", linewidth=0.8)
    ax.set_axis_off()
    fig.text(70.0 / A4_W_MM, 68.0 / A4_H_MM, "controle 50 mm", ha="center", fontsize=7)

    fig.text(
        105.0 / A4_W_MM,
        30.0 / A4_H_MM,
        "Imprimer a 100%, sans ajustement a la page.",
        ha="center",
        fontsize=8,
    )

    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
