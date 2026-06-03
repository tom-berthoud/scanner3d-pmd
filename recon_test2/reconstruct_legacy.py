#!/usr/bin/env python3
"""Reconstruction STL historique depuis les PLY de test.

Cette version est volontairement separee de ``recon_tests`` pour ne pas
interferer avec les essais en cours. Elle reprend l'ancien maillage
``mesh_mode: cylindrical`` du commit ffe8f13: profils ordonnes, grille
verticale commune, quads entre profils voisins, puis capuchons haut/bas.

Usage:
    python recon_test2/reconstruct_legacy.py
    python recon_test2/reconstruct_legacy.py --profile-mode theta
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_INPUT = PROJECT_ROOT / "recon_tests" / "input"
DEFAULT_OUTPUT = HERE / "output"


def load_xyz_ply(path: Path) -> np.ndarray:
    """Load an ASCII XYZ PLY point cloud as ``(N, 3)`` float64.

    Les PLY presents dans ``recon_tests/input`` sont ASCII et ne contiennent
    que x/y/z. On garde un parseur minimal pour que le banc reste autonome.
    """
    vertex_count: int | None = None
    rows: list[tuple[float, float, float]] = []
    in_body = False

    with path.open("r", encoding="ascii") as fh:
        for line in fh:
            stripped = line.strip()
            if not in_body:
                if stripped.startswith("element vertex "):
                    vertex_count = int(stripped.split()[-1])
                elif stripped == "end_header":
                    in_body = True
                continue

            parts = stripped.split()
            if len(parts) >= 3:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))

    points = np.asarray(rows, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{path}: expected XYZ points, got {points.shape}")
    if vertex_count is not None and points.shape[0] != vertex_count:
        raise ValueError(
            f"{path}: header announces {vertex_count} vertices, read {points.shape[0]}"
        )
    return points


def prepare_cloud(points: np.ndarray, dedupe_decimals: int = 6) -> np.ndarray:
    """Remove non-finite values and optional duplicate XYZ rows."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {cloud.shape}")
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if dedupe_decimals >= 0:
        rounded = cloud.round(decimals=dedupe_decimals)
        _, first_indices = np.unique(rounded, axis=0, return_index=True)
        cloud = cloud[np.sort(first_indices)]
    if cloud.shape[0] < 4:
        raise ValueError("Need at least 4 finite unique points")
    return cloud


def split_profiles_from_order(
    points: np.ndarray,
    target_profiles: int = 200,
    y_reset_mm: float = 8.0,
    min_profile_points: int = 2,
) -> list[np.ndarray]:
    """Rebuild scan profiles from the historical PLY write order.

    L'ancien pipeline faisait ``np.vstack(profiles)`` avant l'export PLY. Les
    profils consecutifs produisent souvent une chute nette de Y quand on passe
    d'une vue a la suivante. On detecte ces resets, puis on ajuste doucement
    vers ``target_profiles`` pour eviter les gros blocs quand certains resets
    ont ete effaces par le filtrage.
    """
    y = points[:, 1]
    reset_indices = np.flatnonzero(np.diff(y) < -float(y_reset_mm)) + 1
    bounds = [0, *reset_indices.tolist(), len(points)]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    segments = _drop_tiny_segments(segments, min_profile_points=min_profile_points)

    if target_profiles > 0:
        segments = _split_to_target_count(
            segments,
            target_profiles=target_profiles,
            min_profile_points=min_profile_points,
        )

    return [points[start:end] for start, end in segments if end - start >= min_profile_points]


def _drop_tiny_segments(
    segments: list[tuple[int, int]],
    min_profile_points: int,
) -> list[tuple[int, int]]:
    """Merge one-point fragments into their nearest neighbour."""
    if not segments:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if end - start >= min_profile_points:
            merged.append((start, end))
            continue
        if merged:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        elif len(segments) > 1:
            next_start, next_end = segments[1]
            merged.append((start, next_end if next_start == end else end))
        else:
            merged.append((start, end))
    return merged


def _split_to_target_count(
    segments: list[tuple[int, int]],
    target_profiles: int,
    min_profile_points: int,
) -> list[tuple[int, int]]:
    """Adjust segment count toward the historical scan-step count."""
    result = list(segments)

    while len(result) > target_profiles:
        lengths = np.array([end - start for start, end in result], dtype=np.int64)
        idx = int(np.argmin(lengths))
        if idx == 0:
            merge_idx = 0
        elif idx == len(result) - 1:
            merge_idx = idx - 1
        else:
            left_len = result[idx - 1][1] - result[idx - 1][0]
            right_len = result[idx + 1][1] - result[idx + 1][0]
            merge_idx = idx - 1 if left_len <= right_len else idx

        start = result[merge_idx][0]
        end = result[merge_idx + 1][1]
        result[merge_idx : merge_idx + 2] = [(start, end)]

    while len(result) < target_profiles:
        lengths = np.array([end - start for start, end in result], dtype=np.int64)
        idx = int(np.argmax(lengths))
        start, end = result[idx]
        length = end - start
        if length < min_profile_points * 2:
            break

        mid = start + length // 2
        result[idx : idx + 1] = [(start, mid), (mid, end)]
    return result


def split_profiles_by_theta(points: np.ndarray, n_theta: int = 200) -> list[np.ndarray]:
    """Alternative: rebuild profiles by polar angle around the Y axis."""
    theta = np.mod(np.arctan2(points[:, 2], points[:, 0]) + 2.0 * np.pi, 2.0 * np.pi)
    bins = np.floor(theta / (2.0 * np.pi) * n_theta).astype(np.int32)
    profiles: list[np.ndarray] = []
    for idx in range(n_theta):
        profile = points[bins == idx]
        if profile.shape[0] >= 2:
            profiles.append(profile)
    return profiles


def _prepare_profile_for_meshing(profile: np.ndarray) -> np.ndarray:
    ordered = np.asarray(profile, dtype=np.float64)
    if ordered.ndim != 2 or ordered.shape[1] != 3 or ordered.shape[0] < 2:
        return np.empty((0, 3), dtype=np.float64)
    order = np.lexsort((ordered[:, 2], ordered[:, 1]))
    return ordered[order]


def _profile_to_unique_y_xyz(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = _prepare_profile_for_meshing(profile)
    if ordered.shape[0] < 2:
        return np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    y_vals = ordered[:, 1]
    rounded = np.round(y_vals, decimals=6)
    unique_y, inverse = np.unique(rounded, return_inverse=True)
    xyz = np.zeros((len(unique_y), 3), dtype=np.float64)
    counts = np.zeros(len(unique_y), dtype=np.int32)
    np.add.at(xyz[:, 0], inverse, ordered[:, 0])
    np.add.at(xyz[:, 1], inverse, ordered[:, 1])
    np.add.at(xyz[:, 2], inverse, ordered[:, 2])
    np.add.at(counts, inverse, 1)
    xyz /= counts[:, np.newaxis]
    return unique_y.astype(np.float64), xyz


def _append_quad_faces(
    faces: list[tuple[int, int, int]],
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    i00: int,
    i10: int,
    i01: int,
    i11: int,
) -> None:
    diag_00_11 = float(np.linalg.norm(p00 - p11))
    diag_10_01 = float(np.linalg.norm(p10 - p01))

    if diag_00_11 <= diag_10_01:
        faces.append((i00, i10, i11))
        faces.append((i00, i11, i01))
    else:
        faces.append((i00, i10, i01))
        faces.append((i10, i11, i01))


def profiles_to_cylindrical_mesh(
    profiles: Sequence[np.ndarray],
    max_y_samples: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the historical cylindrical profile mesh.

    Returns:
        ``(vertices, faces)`` arrays suitable for STL export.
    """
    prepared = [_profile_to_unique_y_xyz(profile) for profile in profiles]
    prepared = [(y_vals, xyz) for y_vals, xyz in prepared if y_vals.size >= 2]
    if len(prepared) < 3:
        raise ValueError("Need at least 3 non-empty profiles to build a cylindrical mesh")

    max_samples = max(len(y_vals) for y_vals, _ in prepared)
    sample_count = max(16, min(max_samples, int(max_y_samples)))
    y_min = min(float(y_vals[0]) for y_vals, _ in prepared)
    y_max = max(float(y_vals[-1]) for y_vals, _ in prepared)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        raise ValueError("Invalid vertical extent for cylindrical meshing")

    y_grid = np.linspace(y_min, y_max, sample_count, dtype=np.float64)
    vertices_list: list[np.ndarray] = []
    grid_indices = np.full((len(prepared), sample_count), -1, dtype=np.int64)

    for profile_idx, (y_vals, xyz) in enumerate(prepared):
        valid_mask = (y_grid >= y_vals[0]) & (y_grid <= y_vals[-1])
        if not np.any(valid_mask):
            continue

        sample_y = y_grid[valid_mask]
        sample_x = np.interp(sample_y, y_vals, xyz[:, 0])
        sample_z = np.interp(sample_y, y_vals, xyz[:, 2])
        sample_xyz = np.column_stack([sample_x, sample_y, sample_z]).astype(np.float64)

        start_idx = len(vertices_list)
        vertices_list.extend(sample_xyz)
        inserted = np.arange(start_idx, start_idx + sample_xyz.shape[0], dtype=np.int64)
        grid_indices[profile_idx, np.flatnonzero(valid_mask)] = inserted

    if not vertices_list:
        raise RuntimeError("Cylindrical mesh contains no vertices")

    vertices = np.asarray(vertices_list, dtype=np.float64)
    faces: list[tuple[int, int, int]] = []

    for profile_idx in range(len(prepared)):
        next_idx = (profile_idx + 1) % len(prepared)
        current = grid_indices[profile_idx]
        nxt = grid_indices[next_idx]

        for row_idx in range(sample_count - 1):
            i00 = int(current[row_idx])
            i01 = int(current[row_idx + 1])
            i10 = int(nxt[row_idx])
            i11 = int(nxt[row_idx + 1])
            if min(i00, i01, i10, i11) < 0:
                continue
            _append_quad_faces(
                faces,
                vertices[i00],
                vertices[i10],
                vertices[i01],
                vertices[i11],
                i00,
                i10,
                i01,
                i11,
            )

    top_ring = [
        int(grid_indices[idx, np.flatnonzero(grid_indices[idx] >= 0)[0]])
        for idx in range(len(prepared))
        if np.any(grid_indices[idx] >= 0)
    ]
    bottom_ring = [
        int(grid_indices[idx, np.flatnonzero(grid_indices[idx] >= 0)[-1]])
        for idx in range(len(prepared))
        if np.any(grid_indices[idx] >= 0)
    ]

    if len(top_ring) >= 3:
        top_center = vertices[top_ring].mean(axis=0)
        top_idx = len(vertices)
        vertices = np.vstack([vertices, top_center])
        for idx in range(len(top_ring)):
            nxt = (idx + 1) % len(top_ring)
            faces.append((top_idx, top_ring[nxt], top_ring[idx]))

    if len(bottom_ring) >= 3:
        bottom_center = vertices[bottom_ring].mean(axis=0)
        bottom_idx = len(vertices)
        vertices = np.vstack([vertices, bottom_center])
        for idx in range(len(bottom_ring)):
            nxt = (idx + 1) % len(bottom_ring)
            faces.append((bottom_idx, bottom_ring[idx], bottom_ring[nxt]))

    if not faces:
        raise RuntimeError("Cylindrical mesh produced no triangles")

    face_array = np.asarray(faces, dtype=np.int64)
    face_array = _remove_degenerate_faces(vertices, face_array)
    if face_array.shape[0] == 0:
        raise RuntimeError("Cylindrical mesh contains no non-degenerate faces")
    return vertices, face_array


def _remove_degenerate_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    areas2 = np.linalg.norm(normals, axis=1)
    return faces[areas2 > 1e-9]


def _triangle_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return normal / norm


def write_ascii_stl(vertices: np.ndarray, faces: np.ndarray, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    solid = _stl_safe_name(name)
    with path.open("w", encoding="ascii") as fh:
        fh.write(f"solid {solid}\n")
        for face in faces:
            p0, p1, p2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            n = _triangle_normal(p0, p1, p2)
            fh.write(f"  facet normal {n[0]:.8e} {n[1]:.8e} {n[2]:.8e}\n")
            fh.write("    outer loop\n")
            fh.write(f"      vertex {p0[0]:.8e} {p0[1]:.8e} {p0[2]:.8e}\n")
            fh.write(f"      vertex {p1[0]:.8e} {p1[1]:.8e} {p1[2]:.8e}\n")
            fh.write(f"      vertex {p2[0]:.8e} {p2[1]:.8e} {p2[2]:.8e}\n")
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
        fh.write(f"endsolid {solid}\n")


def _stl_safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".ply":
            raise ValueError(f"Input file must be .ply: {input_path}")
        yield input_path
        return
    yield from sorted(input_path.glob("*.ply"))


def reconstruct_one(path: Path, args: argparse.Namespace) -> Path:
    raw = load_xyz_ply(path)
    cloud = prepare_cloud(raw, dedupe_decimals=args.dedupe_decimals)

    if args.profile_mode == "theta":
        profiles = split_profiles_by_theta(cloud, n_theta=args.target_profiles)
    else:
        profiles = split_profiles_from_order(
            cloud,
            target_profiles=args.target_profiles,
            y_reset_mm=args.y_reset_mm,
            min_profile_points=args.min_profile_points,
        )

    vertices, faces = profiles_to_cylindrical_mesh(
        profiles,
        max_y_samples=args.max_y_samples,
    )

    out_dir = args.output / args.profile_mode
    out_path = out_dir / f"{path.stem}.stl"
    write_ascii_stl(vertices, faces, out_path, name=path.stem)

    print(
        f"{path.name}: {len(raw)} pts -> {len(profiles)} profils -> "
        f"{len(vertices)} vertices / {len(faces)} faces -> {out_path}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruction STL avec l'ancien maillage cylindrique mono-camera."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"PLY ou dossier de PLY (defaut: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Dossier de sortie (defaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--profile-mode",
        choices=("ordered", "theta"),
        default="ordered",
        help="ordered = ordre historique du PLY; theta = bins angulaires autour de Y",
    )
    parser.add_argument(
        "--target-profiles",
        type=int,
        default=200,
        help="Nombre cible de profils reconstruits (0 = ne pas ajuster)",
    )
    parser.add_argument(
        "--y-reset-mm",
        type=float,
        default=8.0,
        help="Chute de Y utilisee pour detecter un nouveau profil en mode ordered",
    )
    parser.add_argument(
        "--min-profile-points",
        type=int,
        default=2,
        help="Nombre minimum de points pour garder un profil",
    )
    parser.add_argument(
        "--max-y-samples",
        type=int,
        default=1024,
        help="Limite de resolution verticale du maillage",
    )
    parser.add_argument(
        "--dedupe-decimals",
        type=int,
        default=6,
        help="-1 pour desactiver la suppression des doublons arrondis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    files = list(iter_input_files(args.input))
    if not files:
        raise SystemExit(f"Aucun .ply trouve dans {args.input}")

    for path in files:
        reconstruct_one(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
