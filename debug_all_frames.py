"""Répartition des détections laser sur toutes les frames d'un scan.

Usage (sur le Pi, dans le venv):
    python debug_all_frames.py /tmp/scan_frames
"""

import sys
import os
import glob
import numpy as np
import cv2

from scanner.processing.laser_line import extract_laser_line


def main(scan_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(scan_dir, "frame_*.jpg")))
    if not files:
        print(f"Aucune frame trouvée dans {scan_dir}")
        return

    print(f"Analyse de {len(files)} frames — seuil 50, min_pixels 15")
    print()

    counts = []
    empty_frames = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            counts.append(0)
            continue
        line = extract_laser_line(img, threshold=50, min_pixels=15, subpixel=True)
        counts.append(line.shape[0])
        if line.shape[0] == 0:
            empty_frames.append(os.path.basename(path))

    counts = np.array(counts)
    print(f"Frames avec ≥1 détection : {int((counts > 0).sum())}/{len(counts)}")
    print(f"Frames vides (rejetées)   : {int((counts == 0).sum())}/{len(counts)}")
    print()
    print(f"Détections par frame — min: {counts.min()}  max: {counts.max()}  "
          f"mean: {counts.mean():.1f}  median: {np.median(counts):.0f}")
    print(f"Total pixels 3D potentiels : {int(counts.sum())}")
    print()

    # Distribution par tranche de 25 frames (≈ 45°)
    print("Distribution angulaire (25 frames par bloc ≈ 45°):")
    for i in range(0, len(counts), 25):
        bloc = counts[i:i + 25]
        angle_start = int(i * 360 / len(counts))
        angle_end = int((i + 25) * 360 / len(counts))
        avg = bloc.mean()
        zeros = int((bloc == 0).sum())
        bar = "#" * int(avg / 2)
        print(f"  {angle_start:3d}°-{angle_end:3d}°  avg={avg:5.1f}  vides={zeros:2d}  {bar}")

    if empty_frames[:5]:
        print()
        print(f"Exemples frames vides : {empty_frames[:5]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/scan_frames")
