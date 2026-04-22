"""Trace la ligne laser détectée (colonnes avec pixel G > seuil).

Usage:
    python debug_line_trace.py <image.jpg> [seuil]

Affiche toutes les colonnes où G >= seuil, avec position verticale et valeur.
"""

import sys
import cv2
import numpy as np


def main(path: str, threshold: int = 50) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"Image illisible: {path}")
        return

    h, w = img.shape[:2]
    G = img[:, :, 1]

    print(f"Image {w}x{h}  |  seuil G>={threshold}")
    print(f"Canaux max : B={img[:,:,0].max()} G={img[:,:,1].max()} R={img[:,:,2].max()}")
    print()

    # Pour chaque colonne, trouver la position et valeur du pixel le plus brillant
    print("Toutes les colonnes avec ≥1 pixel G>=seuil :")
    print(f"{'col':>4} {'row_max':>7} {'G_max':>5} {'row_min':>7} {'row_max_row':>11} {'n_pixels_above':>14}")
    count = 0
    detected_rows = []
    for col in range(w):
        col_g = G[:, col]
        mask = col_g >= threshold
        if not mask.any():
            continue
        count += 1
        row_max = int(col_g.argmax())
        g_max = int(col_g.max())
        above = np.where(mask)[0]
        row_first = int(above.min())
        row_last = int(above.max())
        n_above = len(above)
        detected_rows.extend([row_first, row_last])
        if count <= 50 or col % 20 == 0:
            print(f"{col:>4} {row_max:>7} {g_max:>5} {row_first:>7} {row_last:>11} {n_above:>14}")

    print()
    print(f"Colonnes totales détectées : {count} / {w}")
    if detected_rows:
        print(f"Étendue verticale globale : rows {min(detected_rows)} à {max(detected_rows)} "
              f"(= {max(detected_rows) - min(detected_rows)} px = "
              f"{100*(max(detected_rows) - min(detected_rows))/h:.1f}% de l'image)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/frame.jpg"
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    main(path, t)
