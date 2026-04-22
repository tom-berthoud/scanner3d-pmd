"""Diagnostic de l'extraction laser sur une frame capturée.

Usage (sur le Pi, dans le venv):
    python debug_extraction.py /tmp/scan_frames/frame_050.jpg

Ou pour tester toutes les frames:
    python debug_extraction.py /tmp/scan_frames
"""

import sys
import os
import glob
import numpy as np
import cv2

from scanner.processing.laser_line import extract_laser_line


def analyse_frame(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"[{path}] ❌ image illisible")
        return

    h, w = img.shape[:2]
    B = img[:, :, 0].astype(int)
    G = img[:, :, 1].astype(int)
    R = img[:, :, 2].astype(int)
    signal = np.clip(G - np.maximum(R, B), 0, 255)

    # Statistiques globales
    saturated = int(((R == 255) & (G == 255) & (B == 255)).sum())
    very_bright_green = int((G > 200).sum())

    print(f"\n=== {os.path.basename(path)} ({w}x{h}) ===")
    print(f"  B mean={B.mean():.1f}  G mean={G.mean():.1f}  R mean={R.mean():.1f}")
    print(f"  G-max(R,B) : max={signal.max()}  mean={signal.mean():.2f}")
    print(f"  pixels saturés (255,255,255) : {saturated}")
    print(f"  pixels G>200 : {very_bright_green}")

    # Test extraction à plusieurs seuils
    for t in [10, 30, 60, 100, 150, 180]:
        line = extract_laser_line(img, threshold=t, min_pixels=3)
        marker = " ← actuel (seuil 180)" if t == 180 else ""
        print(f"  seuil {t:3d} → {line.shape[0]:4d} colonnes détectées{marker}")

    # Verdict
    if signal.max() < 30:
        print("  ⚠️  Signal vert dominant trop faible — le laser n'est probablement pas détecté")
        print("      Causes probables : laser éteint, image en RGB au lieu de BGR,")
        print("                          ou laser saturé (overexposure)")
    elif saturated > 1000:
        print("  ⚠️  Beaucoup de pixels blancs saturés — baisser exposure_us")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "frame_*.jpg")))
        # Analyser 5 frames réparties
        if files:
            indices = [0, len(files) // 4, len(files) // 2, 3 * len(files) // 4, len(files) - 1]
            for i in indices:
                analyse_frame(files[i])
        else:
            print(f"Aucune frame trouvée dans {target}")
    else:
        analyse_frame(target)
