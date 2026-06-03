#!/usr/bin/env python3
"""Rendu PNG headless (EGL) d'un mesh ou nuage, vues multiples, pour
vérifier visuellement une reconstruction sans viewer interactif."""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def _bounds(geom) -> tuple[np.ndarray, float]:
    bb = geom.get_axis_aligned_bounding_box()
    center = bb.get_center()
    extent = np.asarray(bb.get_extent())
    return center, float(np.linalg.norm(extent))


def render_views(geom, out_png: str, title: str = "", point_size: float = 3.0) -> str:
    """Rend 4 vues (face, côté, dessus, iso) côte à côte dans un PNG."""
    w = h = 480
    r = rendering.OffscreenRenderer(w, h)
    r.scene.set_background([1, 1, 1, 1])
    r.scene.scene.set_sun_light([-0.3, -0.5, -0.8], [1, 1, 1], 75000)
    r.scene.scene.enable_sun_light(True)
    try:
        r.scene.scene.set_indirect_light_intensity(45000)
    except Exception:
        pass

    is_mesh = isinstance(geom, o3d.geometry.TriangleMesh)
    mat = rendering.MaterialRecord()
    if is_mesh:
        geom.compute_triangle_normals()
        geom.orient_triangles()  # cohérence des faces -> pas de zones noires
        geom.compute_vertex_normals()
        mat.shader = "defaultLit"
        mat.base_color = [0.72, 0.74, 0.8, 1.0]
    else:
        mat.shader = "defaultUnlit"
        mat.point_size = point_size
        mat.base_color = [0.1, 0.3, 0.7, 1.0]
    r.scene.add_geometry("g", geom, mat)

    center, diag = _bounds(geom)
    dist = diag * 1.2
    # (eye direction) pour 4 vues ; Y vertical
    dirs = {
        "face": [0, 0, 1],
        "cote": [1, 0, 0],
        "dessus": [0, 1, 0.001],
        "iso": [0.8, 0.6, 0.8],
    }
    tiles = []
    for name, d in dirs.items():
        d = np.asarray(d, float)
        eye = center + d / np.linalg.norm(d) * dist
        up = [0, 0, 1] if name == "dessus" else [0, 1, 0]
        r.scene.camera.look_at(center, eye, up)
        img = r.render_to_image()
        tiles.append((name, np.asarray(img)))

    # assemble 2x2 avec bandeau titre
    row1 = np.concatenate([tiles[0][1], tiles[1][1]], axis=1)
    row2 = np.concatenate([tiles[2][1], tiles[3][1]], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(grid)
        d = ImageDraw.Draw(im)
        labels = [("face", 4, 4), ("cote", w + 4, 4), ("dessus", 4, h + 4), ("iso", w + 4, h + 4)]
        for t, x, y in labels:
            d.text((x, y), t, fill=(200, 0, 0))
        if title:
            d.text((4, 2 * h - 16), title, fill=(0, 0, 0))
        im.save(out_png)
    except Exception:
        o3d.io.write_image(out_png, o3d.geometry.Image(grid))
    return out_png


def main() -> None:
    import argparse
    from glob import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help=".ply / .stl à rendre (globs ok)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "renders"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    files: list[str] = []
    for p in args.paths:
        files.extend(sorted(glob(os.path.expanduser(p))))
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        geom = o3d.io.read_triangle_mesh(f) if ext in (".stl", ".obj", ".ply") and ext != ".ply" \
            else o3d.io.read_point_cloud(f)
        if ext == ".ply" and len(np.asarray(geom.points)) == 0:
            geom = o3d.io.read_triangle_mesh(f)
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(args.outdir, name + ".png")
        render_views(geom, out, title=name)
        print("->", out)


if __name__ == "__main__":
    main()
