"""scanner.export.stl — Export point cloud as STL or OBJ mesh.

Converts a 3D point cloud to a watertight mesh using trimesh's convex hull
(fast and reliable) with an optional alpha-shape pass for concave objects.
"""

import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def export_stl(
    cloud: np.ndarray,
    path: str,
    profiles: Sequence[np.ndarray] | None = None,
    mesh_mode: str = "cylindrical",
    alpha: float | None = None,
) -> None:
    """Export *cloud* as a binary STL file at *path*.

    The mesh can be built from ordered scan profiles or from the merged
    point cloud. When ordered profiles are available, ``mesh_mode='profiles'``
    or ``'auto'`` will build a strip mesh between consecutive profiles and
    close it with top/bottom caps. Otherwise the exporter falls back to the
    legacy point-cloud meshing path.

    Args:
        cloud: Float array of shape (N, 3) — point cloud in mm.
        path: Destination file path (must end with .stl).
        profiles: Optional ordered 3-D profiles, one per scan angle.
        mesh_mode: ``cylindrical``, ``profiles``, ``cloud`` or ``auto``.
        alpha: Optional alpha-shape radius in mm when ``mesh_mode='cloud'``.
            ``None`` keeps the automatic heuristic.

    Raises:
        ValueError: if *cloud* has fewer than 4 points.
        RuntimeError: if mesh construction fails.
    """
    mesh = _build_mesh(cloud, profiles=profiles, mesh_mode=mesh_mode, alpha=alpha)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.export(path, file_type="stl")
    file_size = os.path.getsize(path)
    logger.info(
        "export_stl: saved %d vertices / %d faces to %s (%.1f KB)",
        len(mesh.vertices),
        len(mesh.faces),
        path,
        file_size / 1024.0,
    )


def export_obj(
    cloud: np.ndarray,
    path: str,
    profiles: Sequence[np.ndarray] | None = None,
    mesh_mode: str = "cylindrical",
    alpha: float | None = None,
) -> None:
    """Export *cloud* as a Wavefront OBJ file at *path*.

    Args:
        cloud: Float array of shape (N, 3) — point cloud in mm.
        path: Destination file path (must end with .obj).
        profiles: Optional ordered 3-D profiles, one per scan angle.
        mesh_mode: ``cylindrical``, ``profiles``, ``cloud`` or ``auto``.
        alpha: Optional alpha-shape radius in mm when ``mesh_mode='cloud'``.
            ``None`` keeps the automatic heuristic.

    Raises:
        ValueError: if *cloud* has fewer than 4 points.
        RuntimeError: if mesh construction fails.
    """
    mesh = _build_mesh(cloud, profiles=profiles, mesh_mode=mesh_mode, alpha=alpha)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.export(path, file_type="obj")
    file_size = os.path.getsize(path)
    logger.info(
        "export_obj: saved %d vertices / %d faces to %s (%.1f KB)",
        len(mesh.vertices),
        len(mesh.faces),
        path,
        file_size / 1024.0,
    )


def _build_mesh(
    cloud: np.ndarray,
    profiles: Sequence[np.ndarray] | None = None,
    mesh_mode: str = "cylindrical",
    alpha: float | None = None,
) -> "trimesh.Trimesh":  # type: ignore[name-defined]
    if mesh_mode not in {"auto", "cylindrical", "profiles", "cloud"}:
        raise ValueError(f"Unknown mesh_mode {mesh_mode!r}")

    if mesh_mode in {"auto", "cylindrical"} and profiles:
        try:
            return _profiles_to_cylindrical_mesh(list(profiles))
        except Exception as exc:
            if mesh_mode == "cylindrical":
                raise RuntimeError(f"Cylindrical mesh construction failed: {exc}") from exc
            logger.debug("Cylindrical mesh failed: %s — trying profile strips", exc)

    if mesh_mode in {"auto", "profiles"} and profiles:
        try:
            return _profiles_to_mesh(list(profiles))
        except Exception as exc:
            if mesh_mode == "profiles":
                raise RuntimeError(f"Profile-strip mesh construction failed: {exc}") from exc
            logger.debug("Profile-strip mesh failed: %s — falling back to cloud mesh", exc)

    return _cloud_to_mesh(cloud, alpha=alpha)


def _cloud_to_mesh(
    cloud: np.ndarray,
    alpha: float | None = None,
) -> "trimesh.Trimesh":  # type: ignore[name-defined]
    """Convert a point cloud to a trimesh mesh.

    Attempts alpha shape first; falls back to convex hull.

    Args:
        cloud: Float array (N, 3).
        alpha: Optional alpha-shape radius in mm. ``None`` uses an automatic
            heuristic based on the bounding-box diagonal.

    Returns:
        A trimesh.Trimesh instance.

    Raises:
        ValueError: if *cloud* has fewer than 4 non-degenerate points.
    """
    try:
        import trimesh  # type: ignore[import]
        import trimesh.creation  # type: ignore[import]
        import trimesh.repair  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("trimesh not available — install with: pip install trimesh") from exc

    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3), got {cloud.shape}")

    n = cloud.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 points to build a mesh, got {n}")

    # Remove duplicate points
    cloud_unique = np.unique(cloud.round(decimals=4), axis=0)
    if cloud_unique.shape[0] < 4:
        raise ValueError("Too few unique points after deduplication")

    def _prepare_alpha_mesh(candidate: "trimesh.Trimesh") -> "trimesh.Trimesh | None":  # type: ignore[name-defined]
        if candidate is None or len(candidate.faces) == 0:
            return None

        mesh_candidate = candidate.copy()
        try:
            components = mesh_candidate.split(only_watertight=False)
            if len(components) > 1:
                mesh_candidate = max(components, key=lambda comp: len(comp.faces))
        except Exception:
            pass

        try:
            mesh_candidate.remove_duplicate_faces()
        except Exception:
            pass
        try:
            mesh_candidate.remove_degenerate_faces()
        except Exception:
            pass
        try:
            mesh_candidate.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            trimesh.repair.fill_holes(mesh_candidate)
        except Exception:
            pass
        try:
            trimesh.repair.fix_normals(mesh_candidate)
        except Exception:
            pass

        if len(mesh_candidate.faces) == 0:
            return None
        return mesh_candidate

    # Try alpha shape (better for concave objects)
    mesh = None
    try:
        import trimesh.creation  # noqa: F811

        if alpha is None:
            bbox_diag = float(np.linalg.norm(cloud_unique.max(axis=0) - cloud_unique.min(axis=0)))
            alpha_value = bbox_diag / 10.0
        else:
            alpha_value = float(alpha)
            if alpha_value <= 0.0:
                raise ValueError(f"alpha must be > 0, got {alpha_value}")

        alpha_mesh = trimesh.creation.icosphere()  # placeholder
        # Use trimesh's alpha_shape if available
        if hasattr(trimesh, "creation") and hasattr(trimesh.creation, "alpha_shape"):
            alpha_mesh = trimesh.creation.alpha_shape(cloud_unique, alpha=alpha_value)
            mesh = _prepare_alpha_mesh(alpha_mesh)
            if mesh is not None:
                logger.debug(
                    "Mesh built via alpha shape (alpha=%.2f, watertight=%s, faces=%d)",
                    alpha_value,
                    mesh.is_watertight,
                    len(mesh.faces),
                )
    except Exception as alpha_exc:
        logger.debug("Alpha shape failed: %s — falling back to convex hull", alpha_exc)

    # Fallback: convex hull (always valid for ≥ 4 non-coplanar points)
    if mesh is None:
        try:
            hull = trimesh.convex.convex_hull(cloud_unique)
            mesh = hull
            logger.debug("Mesh built via convex hull (%d faces)", len(mesh.faces))
        except Exception as hull_exc:
            raise RuntimeError(f"Mesh construction failed: {hull_exc}") from hull_exc

    return mesh


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


def _profile_step_scale(profile: np.ndarray) -> float:
    if profile.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    steps = steps[steps > 1e-9]
    if steps.size == 0:
        return 0.0
    return float(np.median(steps))


def _triangle_is_reasonable(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    local_scale: float,
) -> bool:
    edges = np.array(
        [
            np.linalg.norm(p0 - p1),
            np.linalg.norm(p1 - p2),
            np.linalg.norm(p2 - p0),
        ],
        dtype=np.float64,
    )
    if np.any(edges < 1e-9):
        return False
    if local_scale <= 1e-9:
        return True
    return bool(edges.max() <= local_scale * 6.0)


def _append_profile_pair_faces(
    faces: list[tuple[int, int, int]],
    profile_a: np.ndarray,
    profile_b: np.ndarray,
    offset_a: int,
    offset_b: int,
) -> None:
    if profile_a.shape[0] < 2 or profile_b.shape[0] < 2:
        return

    scale_a = _profile_step_scale(profile_a)
    scale_b = _profile_step_scale(profile_b)
    bridge_samples = min(profile_a.shape[0], profile_b.shape[0], 32)
    if bridge_samples > 0:
        idx_a = np.linspace(0, profile_a.shape[0] - 1, bridge_samples).round().astype(np.int32)
        idx_b = np.linspace(0, profile_b.shape[0] - 1, bridge_samples).round().astype(np.int32)
        bridge = np.linalg.norm(profile_a[idx_a] - profile_b[idx_b], axis=1)
        bridge_scale = float(np.median(bridge))
    else:
        bridge_scale = 0.0
    local_scale = max(scale_a, scale_b, bridge_scale, 1e-6)

    i = 0
    j = 0
    while i < profile_a.shape[0] - 1 and j < profile_b.shape[0] - 1:
        advance_a_cost = float(np.linalg.norm(profile_a[i + 1] - profile_b[j]))
        advance_b_cost = float(np.linalg.norm(profile_a[i] - profile_b[j + 1]))

        if advance_a_cost <= advance_b_cost:
            p0 = profile_a[i]
            p1 = profile_b[j]
            p2 = profile_a[i + 1]
            if _triangle_is_reasonable(p0, p1, p2, local_scale):
                faces.append((offset_a + i, offset_b + j, offset_a + i + 1))
            i += 1
        else:
            p0 = profile_a[i]
            p1 = profile_b[j]
            p2 = profile_b[j + 1]
            if _triangle_is_reasonable(p0, p1, p2, local_scale):
                faces.append((offset_a + i, offset_b + j, offset_b + j + 1))
            j += 1

    while i < profile_a.shape[0] - 1:
        p0 = profile_a[i]
        p1 = profile_b[-1]
        p2 = profile_a[i + 1]
        if _triangle_is_reasonable(p0, p1, p2, local_scale):
            faces.append((offset_a + i, offset_b + profile_b.shape[0] - 1, offset_a + i + 1))
        i += 1

    while j < profile_b.shape[0] - 1:
        p0 = profile_a[-1]
        p1 = profile_b[j]
        p2 = profile_b[j + 1]
        if _triangle_is_reasonable(p0, p1, p2, local_scale):
            faces.append((offset_a + profile_a.shape[0] - 1, offset_b + j, offset_b + j + 1))
        j += 1


def _append_profile_end_caps(
    vertices: np.ndarray,
    faces: list[tuple[int, int, int]],
    ordered_profiles: list[np.ndarray],
    offsets: list[int],
) -> np.ndarray:
    if len(ordered_profiles) < 3:
        return vertices

    top_points = np.array([profile[0] for profile in ordered_profiles], dtype=np.float64)
    bottom_points = np.array([profile[-1] for profile in ordered_profiles], dtype=np.float64)

    top_center = top_points.mean(axis=0)
    bottom_center = bottom_points.mean(axis=0)

    top_center_idx = vertices.shape[0]
    bottom_center_idx = vertices.shape[0] + 1
    vertices = np.vstack([vertices, top_center, bottom_center])

    for idx in range(len(ordered_profiles)):
        next_idx = (idx + 1) % len(ordered_profiles)
        top_a = offsets[idx]
        top_b = offsets[next_idx]
        faces.append((top_center_idx, top_b, top_a))

        bottom_a = offsets[idx] + ordered_profiles[idx].shape[0] - 1
        bottom_b = offsets[next_idx] + ordered_profiles[next_idx].shape[0] - 1
        faces.append((bottom_center_idx, bottom_a, bottom_b))

    return vertices


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


def _profiles_to_cylindrical_mesh(profiles: Sequence[np.ndarray]) -> "trimesh.Trimesh":  # type: ignore[name-defined]
    try:
        import trimesh  # type: ignore[import]
        import trimesh.repair  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("trimesh not available — install with: pip install trimesh") from exc

    prepared = [_profile_to_unique_y_xyz(profile) for profile in profiles]
    prepared = [(y_vals, xyz) for y_vals, xyz in prepared if y_vals.size >= 2]
    if len(prepared) < 3:
        raise ValueError("Need at least 3 non-empty profiles to build a cylindrical mesh")

    max_samples = max(len(y_vals) for y_vals, _ in prepared)
    sample_count = max(16, min(max_samples, 1024))
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

    top_ring = [int(grid_indices[idx, np.flatnonzero(grid_indices[idx] >= 0)[0]]) for idx in range(len(prepared)) if np.any(grid_indices[idx] >= 0)]
    bottom_ring = [int(grid_indices[idx, np.flatnonzero(grid_indices[idx] >= 0)[-1]]) for idx in range(len(prepared)) if np.any(grid_indices[idx] >= 0)]
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

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    if len(mesh.faces) == 0:
        raise RuntimeError("Cylindrical mesh contains no faces after trimesh conversion")

    logger.debug(
        "Mesh built via cylindrical profile surface (%d vertices, %d faces)",
        len(mesh.vertices),
        len(mesh.faces),
    )
    return mesh


def _profiles_to_mesh(profiles: Sequence[np.ndarray]) -> "trimesh.Trimesh":  # type: ignore[name-defined]
    try:
        import trimesh  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("trimesh not available — install with: pip install trimesh") from exc

    ordered_profiles = [_prepare_profile_for_meshing(profile) for profile in profiles]
    ordered_profiles = [profile for profile in ordered_profiles if profile.shape[0] >= 2]
    if len(ordered_profiles) < 2:
        raise ValueError("Need at least 2 non-empty profiles to build a strip mesh")

    vertices = np.vstack(ordered_profiles).astype(np.float64)
    offsets: list[int] = []
    cursor = 0
    for profile in ordered_profiles:
        offsets.append(cursor)
        cursor += profile.shape[0]

    faces: list[tuple[int, int, int]] = []
    for idx in range(len(ordered_profiles)):
        next_idx = (idx + 1) % len(ordered_profiles)
        _append_profile_pair_faces(
            faces,
            ordered_profiles[idx],
            ordered_profiles[next_idx],
            offsets[idx],
            offsets[next_idx],
        )

    vertices = _append_profile_end_caps(vertices, faces, ordered_profiles, offsets)

    if not faces:
        raise RuntimeError("Profile-strip mesh produced no triangles")

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)
    if len(mesh.faces) == 0:
        raise RuntimeError("Profile-strip mesh contains no faces after trimesh conversion")
    logger.debug(
        "Mesh built via profile strips (%d vertices, %d faces)",
        len(mesh.vertices),
        len(mesh.faces),
    )
    return mesh
