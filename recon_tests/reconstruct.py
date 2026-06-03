#!/usr/bin/env python3
"""Banc de test : reconstruire des STL depuis des nuages de points PLY (XYZ)
avec plusieurs méthodes, pour comparer.

Usage:
    python reconstruct.py                 # tous les .ply de ./input, toutes les méthodes
    python reconstruct.py --input ~/Téléchargements --methods cylindrical,poisson_radial
    python reconstruct.py --no-clean      # désactive le nettoyage du coeur parasite

Sorties: recon_tests/output/<methode>/<nom>.stl  + un tableau récapitulatif.

L'axe de rotation du plateau est Y (vertical), comme dans le scanner.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from glob import glob

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
VERTICAL_AXIS = 1  # Y


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_xyz_ply(path: str) -> np.ndarray:
    """Charge un PLY ascii/binaire en (N,3) float64 via Open3D, avec repli ascii."""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size:
        return pts
    # repli: parsing ascii minimal
    rows, in_body = [], False
    with open(path) as fh:
        for line in fh:
            if not in_body:
                if line.strip() == "end_header":
                    in_body = True
                continue
            p = line.split()
            if len(p) >= 3:
                rows.append((float(p[0]), float(p[1]), float(p[2])))
    return np.asarray(rows, dtype=np.float64)


def to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #
def clean_cloud(points: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Retire le coeur parasite (axe/plateau) + outliers statistiques.

    Heuristique du coeur : on regarde la distribution des rayons (distance à
    l'axe Y). Le plateau/axe crée un pic à petit rayon, séparé de l'objet.
    On retire les points sous un plancher radial adaptatif, puis un
    Statistical Outlier Removal d'Open3D.
    """
    P = points
    ax = [i for i in range(3) if i != VERTICAL_AXIS]
    r = np.hypot(P[:, ax[0]], P[:, ax[1]])

    # Plancher radial adaptatif : 25 % du rayon médian de l'objet, borné.
    # (les points du vrai objet sont à grand rayon ; le coeur est concentré bas.)
    r_med = np.median(r)
    floor = min(max(0.15 * r_med, 1.5), 0.5 * r_med)
    keep = r >= floor
    P2 = P[keep]
    removed_core = int((~keep).sum())

    # Statistical outlier removal
    pcd = to_pcd(P2)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    P3 = np.asarray(pcd.points)

    if verbose:
        print(
            f"    clean: {len(P)} -> {len(P3)} pts "
            f"(coeur r<{floor:.1f}mm: -{removed_core}, outliers: -{len(P2) - len(P3)})"
        )
    return P3


# --------------------------------------------------------------------------- #
# Normales
# --------------------------------------------------------------------------- #
def estimate_normals_radial(pcd: o3d.geometry.PointCloud, radius_mm: float = 8.0) -> None:
    """Estime les normales puis les oriente radialement vers l'extérieur de
    l'axe Y. Beaucoup plus robuste que orient_normals_consistent_tangent_plane
    pour un scan sur tour, car l'orientation 'dehors' est connue a priori."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_mm, max_nn=30)
    )
    pcd.normalize_normals()
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    ax = [i for i in range(3) if i != VERTICAL_AXIS]
    radial = np.zeros_like(P)
    radial[:, ax[0]] = P[:, ax[0]]
    radial[:, ax[1]] = P[:, ax[1]]
    nrm = np.linalg.norm(radial, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    radial /= nrm
    flip = np.sum(N * radial, axis=1) < 0
    N[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(N)


# --------------------------------------------------------------------------- #
# Méthodes de reconstruction -> renvoient un TriangleMesh
# --------------------------------------------------------------------------- #
def m_poisson(points: np.ndarray, depth: int = 8) -> o3d.geometry.TriangleMesh:
    """Poisson 'à l'ancienne' : normales orientées par tangent-plane (fragile)."""
    pcd = to_pcd(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=8.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(min(50, len(points) - 1))
    pcd.normalize_normals()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1
    )
    _trim_density(mesh, dens, 0.02)
    return mesh


def m_poisson_radial(points: np.ndarray, depth: int = 8) -> o3d.geometry.TriangleMesh:
    """Poisson avec normales orientées radialement (recommandé)."""
    pcd = to_pcd(points)
    estimate_normals_radial(pcd)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1
    )
    _trim_density(mesh, dens, 0.02)
    return mesh


def m_bpa(points: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Ball Pivoting : interpole les points réels, n'invente pas de surface."""
    pcd = to_pcd(points)
    estimate_normals_radial(pcd)
    # rayons des billes dérivés de l'espacement moyen
    dists = pcd.compute_nearest_neighbor_distance()
    avg = float(np.mean(dists))
    radii = [avg * k for k in (1.0, 2.0, 3.0, 5.0)]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh


def m_alpha(points: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Alpha shape : enveloppe non convexe, bon si densité homogène."""
    pcd = to_pcd(points)
    dists = pcd.compute_nearest_neighbor_distance()
    alpha = float(np.mean(dists)) * 4.0
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    return mesh


def m_cylindrical(
    points: np.ndarray, n_theta: int = 240, n_y: int | None = None
) -> o3d.geometry.TriangleMesh:
    """Maillage cylindrique structuré (height-field 2.5D).

    Pour chaque secteur angulaire (theta) et chaque tranche (y), garde le
    rayon du point le plus à l'extérieur, puis tisse une grille de quads qui
    fait le tour. Ferme le haut et le bas avec un sommet central.
    Idéal pour objets star-convexes sur plateau tournant.
    """
    P = points
    ax = [i for i in range(3) if i != VERTICAL_AXIS]
    x, z, y = P[:, ax[0]], P[:, ax[1]], P[:, VERTICAL_AXIS]
    r = np.hypot(x, z)
    theta = np.arctan2(z, x)  # [-pi, pi]

    y_min, y_max = y.min(), y.max()
    if n_y is None:
        # ~ une rangée par tranche laser réelle
        n_y = max(8, min(200, int((y_max - y_min) / 1.5)))

    ti = np.clip(((theta + np.pi) / (2 * np.pi) * n_theta).astype(int), 0, n_theta - 1)
    yi = np.clip(((y - y_min) / (y_max - y_min + 1e-9) * n_y).astype(int), 0, n_y - 1)

    # rayon = max par cellule (surface extérieure)
    R = np.full((n_y, n_theta), np.nan)
    for cy, ct, cr in zip(yi, ti, r):
        if np.isnan(R[cy, ct]) or cr > R[cy, ct]:
            R[cy, ct] = cr

    # bouche les trous angulaires par interpolation circulaire par rangée
    for j in range(n_y):
        row = R[j]
        if np.all(np.isnan(row)):
            continue
        idx = np.where(~np.isnan(row))[0]
        if len(idx) == n_theta:
            continue
        full = np.arange(n_theta)
        ext_idx = np.concatenate([idx, idx[:1] + n_theta])
        ext_val = np.concatenate([row[idx], row[idx[:1]]])
        R[j] = np.interp(full, ext_idx, ext_val, period=n_theta)

    # rangées entièrement vides -> moyenne des voisines (rare)
    for j in range(n_y):
        if np.all(np.isnan(R[j])):
            up = next((k for k in range(j + 1, n_y) if not np.all(np.isnan(R[k]))), None)
            dn = next((k for k in range(j - 1, -1, -1) if not np.all(np.isnan(R[k]))), None)
            if up is not None and dn is not None:
                R[j] = 0.5 * (R[up] + R[dn])
            elif up is not None:
                R[j] = R[up]
            elif dn is not None:
                R[j] = R[dn]

    ys = y_min + (np.arange(n_y) + 0.5) / n_y * (y_max - y_min)
    th = -np.pi + (np.arange(n_theta) + 0.5) / n_theta * (2 * np.pi)
    cos_t, sin_t = np.cos(th), np.sin(th)

    verts: list[list[float]] = []
    for j in range(n_y):
        for i in range(n_theta):
            rr = R[j, i]
            v = [0.0, 0.0, 0.0]
            v[ax[0]] = rr * cos_t[i]
            v[ax[1]] = rr * sin_t[i]
            v[VERTICAL_AXIS] = ys[j]
            verts.append(v)

    def vid(j: int, i: int) -> int:
        return j * n_theta + (i % n_theta)

    tris: list[list[int]] = []
    for j in range(n_y - 1):
        for i in range(n_theta):
            a, b = vid(j, i), vid(j, i + 1)
            c, d = vid(j + 1, i), vid(j + 1, i + 1)
            tris.append([a, c, b])
            tris.append([b, c, d])

    # capuchons haut/bas
    bottom_c = len(verts)
    cb = [0.0, 0.0, 0.0]
    cb[VERTICAL_AXIS] = ys[0]
    verts.append(cb)
    top_c = len(verts)
    ct = [0.0, 0.0, 0.0]
    ct[VERTICAL_AXIS] = ys[-1]
    verts.append(ct)
    for i in range(n_theta):
        tris.append([bottom_c, vid(0, i + 1), vid(0, i)])  # bas (vers -Y)
        tris.append([top_c, vid(n_y - 1, i), vid(n_y - 1, i + 1)])  # haut (vers +Y)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    return mesh


def _ray_segment_radius(theta: np.ndarray, ax_pts: np.ndarray) -> np.ndarray:
    """r(theta) par intersection rayon (depuis l'axe) / segments du contour.

    Interpole en CARTÉSIEN (le segment entre deux points d'une face plate EST
    la face), ce qui préserve faces droites et coins, contrairement au binning
    du rayon. NaN si pas d'intersection valable dans un segment encadrant.
    """
    phi = np.arctan2(ax_pts[:, 1], ax_pts[:, 0])
    order = np.argsort(phi)
    phi = phi[order]
    pts = ax_pts[order]
    phi_ext = np.concatenate([phi, phi[:1] + 2 * np.pi])
    pts_ext = np.concatenate([pts, pts[:1]], axis=0)

    out = np.full(theta.shape, np.nan)
    th = np.mod(theta + np.pi, 2 * np.pi) - np.pi
    idx = np.clip(np.searchsorted(phi_ext, th, side="right") - 1, 0, len(phi_ext) - 2)
    A, B = pts_ext[idx], pts_ext[idx + 1]
    d = B - A
    ux, uy = np.cos(theta), np.sin(theta)
    cross_ud = ux * d[:, 1] - uy * d[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (A[:, 0] * d[:, 1] - A[:, 1] * d[:, 0]) / cross_ud
        t = -(ux * A[:, 1] - uy * A[:, 0]) / cross_ud
    good = np.isfinite(s) & (s > 0) & np.isfinite(t) & (t >= -1e-6) & (t <= 1 + 1e-6)
    out[good] = s[good]
    return out


def _circular_median(x: np.ndarray, w: int) -> np.ndarray:
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


def _bilateral_grid(
    R: np.ndarray, win: int, sigma_s: float, sigma_r: float
) -> np.ndarray:
    """Filtre bilatéral sur la grille R(y, θ) : lisse le bruit DANS les faces,
    préserve les arêtes (coins, marches). θ est circulaire (roll), y borné.

    sigma_r (mm) règle la préservation d'arête : un saut de rayon > ~2·sigma_r
    (vraie arête) n'est pas lissé ; le bruit (< sigma_r) l'est.
    """
    ny, nt = R.shape
    acc = np.zeros_like(R)
    wsum = np.zeros_like(R)
    yidx = np.arange(ny)
    tidx = np.arange(nt)
    for dy in range(-win, win + 1):
        ys = np.clip(yidx + dy, 0, ny - 1)
        for dt in range(-win, win + 1):
            shifted = R[ys][:, (tidx + dt) % nt]
            ws = np.exp(-(dy * dy + dt * dt) / (2.0 * sigma_s * sigma_s))
            wr = np.exp(-((shifted - R) ** 2) / (2.0 * sigma_r * sigma_r))
            w = ws * wr
            acc += w * shifted
            wsum += w
    return acc / np.where(wsum == 0, 1.0, wsum)


def _outer_rim(ax_pts: np.ndarray, nbin: int, margin: float) -> np.ndarray:
    """Garde le rebord EXTÉRIEUR d'une tranche : par secteur angulaire, on
    conserve tout ce qui est à moins de *margin* (mm) du rayon max du secteur,
    et on jette les points bien plus internes.

    - mur fin (tranche latérale) : étalement radial < margin → tous gardés →
      faces planes préservées.
    - disque plein (face horizontale dessus/dessous) : seul le rebord est
      gardé → plus de chaos ni de créneaux aux capuchons.
    Le rayon max par secteur est pris au 95e percentile (robuste aux pics).
    """
    if len(ax_pts) < nbin:
        return ax_pts
    phi = np.arctan2(ax_pts[:, 1], ax_pts[:, 0])
    rad = np.hypot(ax_pts[:, 0], ax_pts[:, 1])
    b = np.clip(((phi + np.pi) / (2 * np.pi) * nbin).astype(int), 0, nbin - 1)
    keep = np.zeros(len(ax_pts), dtype=bool)
    for s in range(nbin):
        idx = np.where(b == s)[0]
        if idx.size == 0:
            continue
        rmax = np.percentile(rad[idx], 95.0)
        keep[idx[rad[idx] >= rmax - margin]] = True
    return ax_pts[keep]


def m_contour(
    points: np.ndarray,
    n_theta: int = 360,
    n_y: int | None = None,
    median_w: int = 3,
    band_overlap: float = 0.5,
    rim_bins: int = 180,
    rim_margin: float = 3.0,
    smooth: float = 4.0,
    smooth_win: int = 3,
    smooth_sigma_s: float = 2.0,
) -> o3d.geometry.TriangleMesh:
    """Reconstruction par contours 2D + tissage (recommandé pour le tour).

    Pour chaque tranche horizontale : on extrait le rebord extérieur, on
    échantillonne le contour par intersection rayon/segment (faces droites +
    coins préservés), puis on tisse les tranches en quads et on ferme haut/bas.
    Watertight par construction.

    band_overlap: demi-largeur de la bande de points (en pas de tranche).
    rim_bins/rim_percentile: extraction du rebord externe par secteur (enlève la
                  contamination des faces pleines dessus/dessous -> pas de créneaux).
    smooth      : sigma_r (mm) du filtre bilatéral. 0 = off. ~0.8 lisse le bruit
                  résiduel en gardant coins/marches.
    """
    ax = [i for i in range(3) if i != VERTICAL_AXIS]
    uv, y = points[:, ax], points[:, VERTICAL_AXIS]
    ymin, ymax = y.min(), y.max()
    if n_y is None:
        n_y = max(8, min(220, int((ymax - ymin) / 1.2)))
    theta = -np.pi + (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi

    # centres des tranches + bandes (potentiellement chevauchantes) en y
    step = (ymax - ymin) / n_y
    centers = ymin + (np.arange(n_y) + 0.5) * step
    half = max(0.5, band_overlap) * step
    order_y = np.argsort(y)
    y_sorted = y[order_y]

    R = np.full((n_y, n_theta), np.nan)
    for j in range(n_y):
        lo = np.searchsorted(y_sorted, centers[j] - half, side="left")
        hi = np.searchsorted(y_sorted, centers[j] + half, side="right")
        sl = uv[order_y[lo:hi]]
        if len(sl) < 8:
            continue
        if rim_bins and rim_bins > 0:
            sl = _outer_rim(sl, nbin=rim_bins, margin=rim_margin)
            if len(sl) < 8:
                continue
        R[j] = _circular_median(_ray_segment_radius(theta, sl), median_w)

    full = np.arange(n_theta)
    for j in range(n_y):
        m = np.isfinite(R[j])
        if m.sum() < 3:
            R[j] = np.nan
            continue
        if not m.all():
            i = np.where(m)[0]
            R[j] = np.interp(
                full,
                np.concatenate([i, i[:1] + n_theta]),
                np.concatenate([R[j][i], R[j][i[:1]]]),
                period=n_theta,
            )
    valid = [j for j in range(n_y) if np.isfinite(R[j]).all()]
    if not valid:
        raise RuntimeError("aucune tranche exploitable")
    for j in range(n_y):
        if not np.isfinite(R[j]).all():
            R[j] = R[min(valid, key=lambda k: abs(k - j))]

    # lissage bilatéral : faces planes, arêtes/marches préservées
    if smooth and smooth > 0:
        R = _bilateral_grid(R, win=smooth_win, sigma_s=smooth_sigma_s, sigma_r=smooth)

    ys = ymin + (np.arange(n_y) + 0.5) / n_y * (ymax - ymin)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    verts = np.zeros((n_y * n_theta, 3))
    for j in range(n_y):
        verts[j * n_theta : (j + 1) * n_theta, ax[0]] = R[j] * cos_t
        verts[j * n_theta : (j + 1) * n_theta, ax[1]] = R[j] * sin_t
        verts[j * n_theta : (j + 1) * n_theta, VERTICAL_AXIS] = ys[j]

    def vid(j, i):
        return j * n_theta + (i % n_theta)

    tris = []
    for j in range(n_y - 1):
        for i in range(n_theta):
            a, b, c, d = vid(j, i), vid(j, i + 1), vid(j + 1, i), vid(j + 1, i + 1)
            tris.append([a, c, b])
            tris.append([b, c, d])
    bc = len(verts)
    cb = np.zeros(3); cb[VERTICAL_AXIS] = ys[0]
    tc = bc + 1
    ct = np.zeros(3); ct[VERTICAL_AXIS] = ys[-1]
    verts = np.vstack([verts, cb, ct])
    for i in range(n_theta):
        tris.append([bc, vid(0, i + 1), vid(0, i)])
        tris.append([tc, vid(n_y - 1, i), vid(n_y - 1, i + 1)])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, np.int32))
    return mesh


METHODS = {
    "poisson": m_poisson,
    "poisson_radial": m_poisson_radial,
    "cylindrical": m_cylindrical,
    "contour": m_contour,
    "bpa": m_bpa,
    "alpha": m_alpha,
}


def fit_error(mesh: o3d.geometry.TriangleMesh, points: np.ndarray) -> tuple[float, float]:
    """Distance point->surface (mm) du nuage au maillage: (moyenne, p95).
    Mesure à quel point le STL colle aux données réelles."""
    scene = o3d.t.geometry.RaycastingScene()
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(tm)
    q = o3d.core.Tensor(points.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    d = scene.compute_distance(q).numpy()
    return float(d.mean()), float(np.percentile(d, 95))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _trim_density(mesh, densities, quantile: float) -> None:
    d = np.asarray(densities)
    if d.size == 0 or quantile <= 0:
        return
    mask = d < np.quantile(d, quantile)
    if mask.any():
        mesh.remove_vertices_by_mask(mask)


def finalize(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default=os.path.join(HERE, "input"),
        help="dossier des .ply (défaut: recon_tests/input)",
    )
    ap.add_argument("--output", default=os.path.join(HERE, "output"))
    ap.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"méthodes séparées par des virgules ({', '.join(METHODS)})",
    )
    ap.add_argument("--no-clean", action="store_true", help="ne pas nettoyer le coeur")
    args = ap.parse_args()

    files = sorted(glob(os.path.join(args.input, "*.ply")))
    if not files:
        print(f"Aucun .ply dans {args.input}", file=sys.stderr)
        return 1
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        print(f"Méthodes inconnues: {bad}. Dispo: {list(METHODS)}", file=sys.stderr)
        return 1

    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        raw = load_xyz_ply(f)
        print(f"\n=== {name}  ({len(raw)} pts) ===")
        pts = raw if args.no_clean else clean_cloud(raw)
        for meth in methods:
            outdir = os.path.join(args.output, meth)
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f"{name}.stl")
            t0 = time.time()
            try:
                mesh = finalize(METHODS[meth](pts))
                o3d.io.write_triangle_mesh(outpath, mesh)
                wt = mesh.is_watertight()
                err_mean, err_p95 = fit_error(mesh, pts)
                dt = time.time() - t0
                nv, nf = len(mesh.vertices), len(mesh.triangles)
                print(
                    f"  {meth:16} {nv:7d} v / {nf:7d} f  "
                    f"wt={str(wt):5}  fit(moy/p95)={err_mean:4.2f}/{err_p95:4.2f}mm  {dt:5.1f}s"
                )
                rows.append((name, meth, nv, nf, wt, f"{err_mean:.2f}", f"{err_p95:.2f}", f"{dt:.1f}"))
            except Exception as e:
                print(f"  {meth:16} ÉCHEC: {e.__class__.__name__}: {e}")
                rows.append((name, meth, 0, 0, False, "ERR", "ERR", "ERR"))

    # récapitulatif
    print("\n\n========== RÉCAPITULATIF ==========")
    print(
        f"{'scan':28} {'méthode':16} {'verts':>8} {'faces':>8} {'wt':>5} "
        f"{'fit_moy':>8} {'fit_p95':>8} {'s':>5}"
    )
    for r in rows:
        print(
            f"{r[0]:28} {r[1]:16} {r[2]:8d} {r[3]:8d} {str(r[4]):>5} "
            f"{r[5]:>8} {r[6]:>8} {r[7]:>5}"
        )
    print(f"\nSTL écrits dans: {args.output}/<méthode>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
