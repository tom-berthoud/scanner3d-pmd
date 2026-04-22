"""Mesure l'étendue verticale des points détectés sur une frame.

Usage:
    python debug_vertical.py /tmp/scan_frames/frame_150.jpg
"""

import sys
import cv2
import numpy as np
from scanner.processing.laser_line import extract_laser_line


def main(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"Image illisible: {path}")
        return

    h, w = img.shape[:2]
    for threshold in [20, 30, 50, 80]:
        line = extract_laser_line(img, threshold=threshold, min_pixels=1, subpixel=True)
        if line.shape[0] == 0:
            print(f"seuil {threshold:3d} : 0 points")
            continue
        rows = line[:, 1]
        row_min = int(rows.min())
        row_max = int(rows.max())
        span = row_max - row_min
        span_pct = 100 * span / h
        print(
            f"seuil {threshold:3d} : {line.shape[0]:3d} points  "
            f"row_min={row_min:3d}  row_max={row_max:3d}  "
            f"étendue verticale={span} px ({span_pct:.1f}% de l'image)"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/scan_frames/frame_150.jpg")
