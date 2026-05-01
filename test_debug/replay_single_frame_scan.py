"""Replay a single laser frame as a full synthetic 200-step scan.

This debug helper reuses the real scanner pipeline modules:
    - scanner.processing.extract_laser_line
    - scanner.processing.triangulate
    - scanner.reconstruction.merge_profiles / filter_outliers
    - scanner.export.export_point_cloud_ply / export_stl / export_obj

The input image is treated as if it had been captured at every rotation
angle of the turntable. This is useful to sanity-check line extraction,
triangulation and export without running a real acquisition.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import cv2  # type: ignore[import]
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scanner.calibration import (  # noqa: E402
    CalibrationError,
    approximate_camera_intrinsics,
    load_camera_calibration,
    load_laser_plane,
)
from scanner.export import export_obj, export_point_cloud_ply, export_stl  # noqa: E402
from scanner.processing import extract_laser_line, triangulate  # noqa: E402
from scanner.reconstruction import filter_outliers, merge_profiles  # noqa: E402

logger = logging.getLogger("test_debug.replay_single_frame_scan")

DEFAULT_FRAME = Path(__file__).resolve().parent / "frame.jpg"
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
README_DEBUG_LASER_PLANE = np.array([0.5, 0.0, 0.866, -259.8], dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Traite une seule image comme si elle representait toutes les frames "
            "d'un scan 360 degres."
        )
    )
    parser.add_argument(
        "--frame",
        type=Path,
        default=DEFAULT_FRAME,
        help=f"Image source a reutiliser pour tous les angles (defaut: {DEFAULT_FRAME})",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS,
        help=f"Fichier settings.yaml a utiliser (defaut: {DEFAULT_SETTINGS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie pour les exports (defaut: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Nombre d'angles synthetiques. Par defaut: valeur de scan.n_steps dans settings.",
    )
    parser.add_argument(
        "--format",
        choices=("stl", "obj"),
        default=None,
        help="Format mesh de sortie. Par defaut: export.default_format dans settings.",
    )
    parser.add_argument(
        "--extract-mode",
        choices=("app", "row-green"),
        default="row-green",
        help=(
            "Mode d'extraction: 'app' reutilise scanner.processing.extract_laser_line, "
            "'row-green' prend 1 point par ligne sur le canal vert brut."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Override du seuil. En mode row-green, il s'applique au canal vert brut.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Override du min_line_pixels.",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default=None,
        help="Plan laser explicite sous la forme 'a,b,c,d'.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Desactive le filtrage statistique des outliers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Active les logs DEBUG.",
    )
    return parser.parse_args()


def _load_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_axis_point() -> np.ndarray | None:
    platform_path = PROJECT_ROOT / "config" / "platform.yaml"
    if not platform_path.exists():
        return None
    try:
        with platform_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        point = data.get("rotation_axis_point_mm")
        if point is None:
            return None
        return np.asarray(point, dtype=np.float64).reshape(3)
    except Exception as exc:  # pragma: no cover - debug helper
        logger.warning("platform.yaml illisible, axe ignore: %s", exc)
        return None


def _parse_plane(raw: str) -> np.ndarray:
    values = [float(part.strip()) for part in raw.split(",")]
    if len(values) != 4:
        raise ValueError(f"Plane must have 4 comma-separated values, got {raw!r}")
    return np.asarray(values, dtype=np.float64)


def _load_camera_model(config: dict, frame_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    calib_cfg = config.get("calibration", {})
    cam_cfg = config.get("camera", {})
    use_checkerboard = bool(calib_cfg.get("use_checkerboard", True))
    focal_scale = float(calib_cfg.get("approx_focal_scale", 1.25))

    frame_resolution = (int(frame_shape[1]), int(frame_shape[0]))

    if use_checkerboard:
        try:
            return load_camera_calibration()
        except CalibrationError as exc:
            logger.warning("Calibration camera absente, fallback en intrinseques approx: %s", exc)

    cfg_resolution = cam_cfg.get("resolution", list(frame_resolution))
    try:
        resolution = (int(cfg_resolution[0]), int(cfg_resolution[1]))
    except Exception:
        resolution = frame_resolution

    if resolution != frame_resolution:
        logger.warning(
            "Resolution settings %s differente de l'image %s, utilisation de la taille image.",
            resolution,
            frame_resolution,
        )
        resolution = frame_resolution

    return approximate_camera_intrinsics(resolution, focal_scale=focal_scale)


def _load_laser_plane_for_debug(cli_plane: str | None) -> np.ndarray:
    if cli_plane:
        plane = _parse_plane(cli_plane)
        logger.warning("Plan laser fourni en CLI: %s", plane.tolist())
        return plane

    try:
        return load_laser_plane()
    except CalibrationError as exc:
        logger.warning(
            "Calibration plan laser absente, fallback sur la geometrie README %s: %s",
            README_DEBUG_LASER_PLANE.tolist(),
            exc,
        )
        return README_DEBUG_LASER_PLANE.copy()


def _extract_line_for_debug(
    frame: np.ndarray,
    threshold: int,
    min_pixels: int,
    subpixel: bool,
    allow_threshold_fallback: bool,
) -> tuple[np.ndarray, int]:
    thresholds = [threshold]
    if allow_threshold_fallback:
        thresholds.extend([180, 150, 120, 100, 80, 60, 40, 30, 20, 15, 10, 5])

    seen: set[int] = set()
    for candidate in thresholds:
        if candidate in seen:
            continue
        seen.add(candidate)
        line_px = extract_laser_line(
            frame,
            threshold=candidate,
            min_pixels=min_pixels,
            subpixel=subpixel,
        )
        if line_px.shape[0] > 0:
            if candidate != threshold:
                logger.warning(
                    "Aucune detection au seuil %d, fallback debug retenu: seuil %d (%d points).",
                    threshold,
                    candidate,
                    line_px.shape[0],
                )
            return line_px, candidate

    return np.empty((0, 2), dtype=np.float32), threshold


def _extract_row_green_line(
    frame: np.ndarray,
    threshold: int,
    min_pixels: int,
    subpixel: bool,
) -> np.ndarray:
    green = frame[:, :, 1]
    points: list[tuple[float, float]] = []

    for row in range(green.shape[0]):
        cols = np.flatnonzero(green[row] >= threshold)
        if cols.size == 0:
            continue

        splits = np.where(np.diff(cols) > 1)[0] + 1
        segments = np.split(cols, splits)
        best_segment = max(
            segments,
            key=lambda seg: (int(green[row, seg].sum()), int(seg.size)),
        )
        if subpixel:
            weights = green[row, best_segment].astype(np.float64)
            weights = np.maximum(weights - float(threshold) + 1.0, 1.0)
            col_center = float(np.average(best_segment.astype(np.float64), weights=weights))
        else:
            col_center = float(
                int(round((int(best_segment[0]) + int(best_segment[-1])) / 2.0))
            )
        points.append((col_center, float(row)))

    if len(points) < min_pixels:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(points, dtype=np.float32)


def _export_overlay_image(
    frame: np.ndarray,
    line_px: np.ndarray,
    overlay_path: Path,
    threshold: int,
    min_pixels: int,
) -> None:
    overlay = frame.copy()

    for col_f, row_f in line_px:
        col = int(round(float(col_f)))
        row = int(round(float(row_f)))
        if 0 <= col < overlay.shape[1] and 0 <= row < overlay.shape[0]:
            cv2.circle(overlay, (col, row), 1, (0, 0, 255), -1)

    if line_px.shape[0] >= 2:
        poly = np.round(line_px).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [poly], isClosed=False, color=(255, 255, 0), thickness=1)

    info_lines = [
        f"threshold={threshold}",
        f"min_pixels={min_pixels}",
        f"detected_points={line_px.shape[0]}",
    ]
    y = 22
    for text in info_lines:
        cv2.putText(
            overlay,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Impossible d'ecrire l'overlay: {overlay_path}")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = _load_settings(args.settings)
    frame = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Image illisible: {args.frame}")

    proc_cfg = config.get("processing", {})
    scan_cfg = config.get("scan", {})
    recon_cfg = config.get("reconstruction", {})
    export_cfg = config.get("export", {})

    n_steps = int(args.steps or scan_cfg.get("n_steps", 200))
    if args.threshold is not None:
        threshold = int(args.threshold)
    elif args.extract_mode == "row-green":
        threshold = 80
    else:
        threshold = int(proc_cfg.get("laser_threshold", 60))
    min_pixels = int(args.min_pixels if args.min_pixels is not None else proc_cfg.get("min_line_pixels", 15))
    subpixel = bool(proc_cfg.get("subpixel", True))
    nb_neighbors = int(recon_cfg.get("outlier_nb_neighbors", 20))
    std_ratio = float(recon_cfg.get("outlier_std_ratio", 2.0))
    mesh_format = str(args.format or export_cfg.get("default_format", "stl")).lower()
    poisson_cfg = export_cfg.get("poisson", {})
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix, dist_coeffs = _load_camera_model(config, frame.shape)
    laser_plane = _load_laser_plane_for_debug(args.plane)
    axis_point = _load_axis_point()

    if args.extract_mode == "row-green":
        line_px = _extract_row_green_line(
            frame,
            threshold=threshold,
            min_pixels=min_pixels,
            subpixel=subpixel,
        )
        used_threshold = threshold
    else:
        line_px, used_threshold = _extract_line_for_debug(
            frame=frame,
            threshold=threshold,
            min_pixels=min_pixels,
            subpixel=subpixel,
            allow_threshold_fallback=args.threshold is None,
        )
    if line_px.shape[0] == 0:
        raise RuntimeError(
            "Aucune ligne laser detectee dans l'image. Ajuste --threshold ou verifie frame.jpg."
        )

    angle_step_rad = 2.0 * math.pi / n_steps
    profiles: list[np.ndarray] = []
    for idx in range(n_steps):
        angle_rad = idx * angle_step_rad
        pts_3d = triangulate(
            line_px,
            camera_matrix,
            dist_coeffs,
            laser_plane,
            angle_rad,
            axis_point=axis_point,
        )
        if pts_3d.shape[0] > 0:
            profiles.append(pts_3d)

    cloud = merge_profiles(profiles)
    if cloud.shape[0] == 0:
        raise RuntimeError("Aucun point 3D genere a partir de la frame fournie.")

    if not args.no_filter and cloud.shape[0] >= 20:
        cloud = filter_outliers(cloud, nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = args.frame.stem
    overlay_path = output_dir / f"{stem}_replay_{timestamp}_overlay.jpg"
    cloud_path = output_dir / f"{stem}_replay_{timestamp}_cloud.ply"
    mesh_path = output_dir / f"{stem}_replay_{timestamp}.{mesh_format}"

    _export_overlay_image(frame, line_px, overlay_path, used_threshold, min_pixels)
    export_point_cloud_ply(cloud, str(cloud_path))
    if mesh_format == "obj":
        export_obj(cloud, str(mesh_path), poisson=poisson_cfg)
    else:
        export_stl(cloud, str(mesh_path), poisson=poisson_cfg)

    print(f"frame          : {args.frame}")
    print(f"extract_mode   : {args.extract_mode}")
    print("mesh_method    : poisson")
    print(f"threshold      : {used_threshold}")
    print(f"points 2D/frame: {line_px.shape[0]}")
    print(f"angles simules : {n_steps}")
    print(f"points 3D total: {cloud.shape[0]}")
    print(f"overlay exporte: {overlay_path}")
    print(f"nuage exporte  : {cloud_path}")
    print(f"mesh exporte   : {mesh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
