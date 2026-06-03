#!/usr/bin/env python3
"""Prototype: reconstruction par contours 2D + tissage (loft).

Idée: l'objet sur le plateau est star-convexe autour de l'axe Y. Pour chaque
tranche horizontale on a un contour fermé (x,z). On échantillonne ce contour
par intersection rayon/segment en CARTÉSIEN (préserve les faces droites et les
coins, contrairement au binning du rayon), puis on tisse les tranches en quads.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

VAX = 1  # axe vertical Y


def load(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def clean(P: np.ndarray) -> np.ndarray:
    ax = [i for i in range(3) if i != VAX]
    r = np.hypot(P[:, ax[0]], P[:, ax[1]])
    rmed = np.median(r)
    floor = min(max(0.15 * rmed, 1.5), 0.5 * rmed)
    P = P[r >= floor]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pcd.points)


def _ray_segment_radius(theta: np.ndarray, ax_pts: np.ndarray) -> np.ndarray:
    """Pour chaque angle theta, rayon de l'intersection du rayon (depuis l'axe)
    avec le polygone fermé défini par ax_pts (Nx2, triés par angle).

    ax_pts: coordonnées 2D (u,v) dans le plan perpendiculaire à l'axe.
    Renvoie r(theta), NaN si pas de segment encadrant exploitable.
    """
    u, v = ax_pts[:, 0], ax_pts[:, 1]
    phi = np.arctan2(v, u)  # [-pi, pi]
    order = np.argsort(phi)
    phi = phi[order]
    pts = ax_pts[order]
    # ferme le polygone par wrap-around
    phi_ext = np.concatenate([phi, phi[:1] + 2 * np.pi])
    pts_ext = np.concatenate([pts, pts[:1]], axis=0)

    out = np.full(theta.shape, np.nan)
    th = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # -> [-pi,pi]
    # index du segment encadrant: phi[i] <= th < phi[i+1]
    idx = np.searchsorted(phi_ext, th, side="right") - 1
    idx = np.clip(idx, 0, len(phi_ext) - 2)

    A = pts_ext[idx]
    B = pts_ext[idx + 1]
    d = B - A
    ux, uy = np.cos(theta), np.sin(theta)
    # résout s*u = A + t*d  -> s = cross(A,d)/cross(u,d)
    cross_ud = ux * d[:, 1] - uy * d[:, 0]
    cross_Ad = A[:, 0] * d[:, 1] - A[:, 1] * d[:, 0]
    cross_uA = ux * A[:, 1] - uy * A[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = cross_Ad / cross_ud          # rayon le long du rayon
        t = -cross_uA / cross_ud         # paramètre le long du segment
    # n'accepte que les vraies intersections DANS le segment (pas d'extrapolation)
    good = np.isfinite(s) & (s > 0) & np.isfinite(t) & (t >= -1e-6) & (t <= 1 + 1e-6)
    out[good] = s[good]
    return out


def _circular_median(x: np.ndarray, w: int = 3) -> np.ndarray:
    """Filtre médian circulaire (enlève les pics, garde les arêtes)."""
    n = len(x)
    if w <= 1 or n < w:
        return x
    half = w // 2
    ext = np.concatenate([x[-half:], x, x[:half]])
    out = np.empty(n)
    for i in range(n):
        win = ext[i : i + w]
        valid = win[np.isfinite(win)]
        out[i] = np.median(valid) if valid.size else np.nan
    return out


def reconstruct_contour(
    P: np.ndarray,
    n_theta: int = 360,
    n_y: int | None = None,
    median_w: int = 3,
) -> o3d.geometry.TriangleMesh:
    ax = [i for i in range(3) if i != VAX]
    uv = P[:, ax]
    y = P[:, VAX]
    ymin, ymax = y.min(), y.max()
    if n_y is None:
        n_y = max(8, min(220, int((ymax - ymin) / 1.2)))

    yi = np.clip(((y - ymin) / (ymax - ymin + 1e-9) * n_y).astype(int), 0, n_y - 1)
    theta = -np.pi + (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi

    R = np.full((n_y, n_theta), np.nan)
    for j in range(n_y):
        sl = uv[yi == j]
        if len(sl) < 8:
            continue
        rj = _ray_segment_radius(theta, sl)
        rj = _circular_median(rj, median_w)
        R[j] = rj

    # comble les angles manquants par interpolation circulaire
    full = np.arange(n_theta)
    for j in range(n_y):
        row = R[j]
        m = np.isfinite(row)
        if m.sum() < 3:
            R[j] = np.nan
            continue
        if not m.all():
            idx = np.where(m)[0]
            ext_i = np.concatenate([idx, idx[:1] + n_theta])
            ext_v = np.concatenate([row[idx], row[idx[:1]]])
            R[j] = np.interp(full, ext_i, ext_v, period=n_theta)

    # rangées vides -> voisines
    valid_rows = [j for j in range(n_y) if np.isfinite(R[j]).all()]
    if not valid_rows:
        raise RuntimeError("aucune tranche exploitable")
    for j in range(n_y):
        if not np.isfinite(R[j]).all():
            nearest = min(valid_rows, key=lambda k: abs(k - j))
            R[j] = R[nearest]

    ys = ymin + (np.arange(n_y) + 0.5) / n_y * (ymax - ymin)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    verts = np.zeros((n_y * n_theta, 3))
    for j in range(n_y):
        verts[j * n_theta : (j + 1) * n_theta, ax[0]] = R[j] * cos_t
        verts[j * n_theta : (j + 1) * n_theta, ax[1]] = R[j] * sin_t
        verts[j * n_theta : (j + 1) * n_theta, VAX] = ys[j]

    def vid(j, i):
        return j * n_theta + (i % n_theta)

    tris = []
    for j in range(n_y - 1):
        for i in range(n_theta):
            a, b, c, d = vid(j, i), vid(j, i + 1), vid(j + 1, i), vid(j + 1, i + 1)
            tris.append([a, c, b])
            tris.append([b, c, d])
    bc = len(verts)
    cb = np.zeros(3); cb[VAX] = ys[0]
    tc = bc + 1
    ct = np.zeros(3); ct[VAX] = ys[-1]
    verts = np.vstack([verts, cb, ct])
    for i in range(n_theta):
        tris.append([bc, vid(0, i + 1), vid(0, i)])
        tris.append([tc, vid(n_y - 1, i), vid(n_y - 1, i + 1)])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, np.int32))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    import sys
    from render import render_views

    path = sys.argv[1] if len(sys.argv) > 1 else \
        "recon_tests/input/scan_20260503_233338_cloud.ply"
    nth = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    P = clean(load(path))
    m = reconstruct_contour(P, n_theta=nth)
    print("verts", len(m.vertices), "faces", len(m.triangles),
          "watertight", m.is_watertight())
    import os
    out = os.path.join("recon_tests/renders", f"DEV_contour_nth{nth}.png")
    render_views(m, out, title=f"contour n_theta={nth}")
    o3d.io.write_triangle_mesh(f"recon_tests/output/contour_dev.stl", m)
    print("->", out)
