"""Diagnostique détaillé du canal vert sur une image.

Usage:
    python debug_green.py <image.jpg>

Montre, pour chaque ligne (colonne) de l'image, le nombre de pixels
au-dessus de plusieurs seuils du canal vert.
"""

import sys
import cv2
import numpy as np


def main(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"Image illisible: {path}")
        return

    h, w = img.shape[:2]
    G = img[:, :, 1]  # canal vert en BGR
    R = img[:, :, 2]
    B = img[:, :, 0]

    print(f"Image: {w}x{h}")
    print(f"G : max={G.max()}  mean={G.mean():.1f}  pixels>100: {int((G > 100).sum())}  >150: {int((G > 150).sum())}")
    print(f"R : max={R.max()}  mean={R.mean():.1f}")
    print(f"B : max={B.max()}  mean={B.mean():.1f}")
    print()

    # Pour chaque seuil, combien de COLONNES ont au moins 1 pixel au-dessus ?
    signal = np.clip(G.astype(int) - np.maximum(R, B).astype(int), 0, 255).astype(np.uint8)
    print("Combien de colonnes contiennent ≥1 pixel avec G-max(R,B) >= seuil :")
    for t in [30, 50, 80, 100, 120, 150, 180, 200]:
        mask = signal >= t
        cols_with_detection = int((mask.any(axis=0)).sum())
        total_px = int(mask.sum())
        print(f"  seuil {t:3d} → {cols_with_detection:3d} colonnes détectées ({total_px} pixels au total)")
    print()

    # Afficher le profil vertical du canal G à 5 colonnes réparties
    print("Profil G le long de 5 colonnes (max par bloc de 40 lignes) :")
    for col in [w // 6, w // 3, w // 2, 2 * w // 3, 5 * w // 6]:
        col_data = G[:, col]
        # Max par bloc pour voir où la ligne est visible
        blocks = [int(col_data[i : i + 40].max()) for i in range(0, h, 40)]
        print(f"  col {col:3d}: {blocks}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
