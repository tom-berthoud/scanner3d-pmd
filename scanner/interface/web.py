"""scanner.interface.web — Flask web interface for the 3D scanner.

Routes:
    GET  /                      — Main page (scan control + 3-D viewer)
    POST /scan/start            — Start scan in background thread
    GET  /scan/status           — JSON status (state, progress, message)
    GET  /scan/download         — Download last exported STL/OBJ
    GET  /scan/stream           — SSE stream for real-time progress
    GET  /calibration           — Calibration page
    POST /calibration/camera    — Upload images + run camera calibration
    POST /calibration/laser     — Run laser plane calibration
    GET  /manual                — Manual hardware control page
    GET  /manual/camera/frame   — Manual live camera frame
    POST /manual/laser          — Manual laser on/off
    POST /manual/motor          — Manual motor jog
    POST /manual/led            — Manual LED control

Run with:
    python -m scanner.interface.web
"""

import base64
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def create_app(config_path: Optional[str] = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_path: Path to settings.yaml.  Defaults to
            config/settings.yaml in the project root.

    Returns:
        Configured Flask application instance.
    """
    template_dir = str(Path(__file__).parent / "templates")
    static_dir = str(Path(__file__).parent / "static")
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload limit

    # ------------------------------------------------------------------ #
    # Load settings
    # ------------------------------------------------------------------ #
    default_cfg_path = str(
        Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
    )
    cfg_file = config_path or default_cfg_path

    settings: dict = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh) or {}
        logger.info("Settings loaded from %s", cfg_file)
    else:
        logger.warning("Settings file not found: %s — using defaults", cfg_file)

    # ------------------------------------------------------------------ #
    # Initialise hardware
    # ------------------------------------------------------------------ #
    from scanner.hardware import init_hardware, HardwareError

    try:
        init_hardware(settings)
    except HardwareError as exc:
        logger.warning("Hardware init failed (running in degraded mode): %s", exc)

    # ------------------------------------------------------------------ #
    # Shared scan state
    # ------------------------------------------------------------------ #
    from scanner.orchestration import ScannerState, StateMachine

    _sm = StateMachine()
    _scan_state: dict = {
        "state": "IDLE",
        "progress": 0,
        "message": "Ready",
        "last_file": None,
        "error": None,
        "artifacts": {},
    }
    _scan_lock = threading.Lock()
    _sse_queue: queue.Queue = queue.Queue(maxsize=50)
    _camera_calib_lock = threading.Lock()
    _camera_calib_session: dict = {
        "sessions": {},
    }
    _extrinsics_turntable_lock = threading.Lock()
    _extrinsics_turntable_session: dict = {
        "sessions": {},
    }
    _laser_calib_lock = threading.Lock()
    _laser_calib_session: dict = {
        "captures": [],
        "last_result": None,
    }

    def _project_path(path: str | None) -> str | None:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return str(Path(__file__).resolve().parent.parent.parent / path)

    def _selected_camera_id(default: str | None = None) -> str:
        from scanner.calibration import camera_configs, default_camera_id

        camera_id = str(request.values.get("camera") or request.values.get("camera_id") or default or "")
        valid_ids = {str(cam_cfg.get("id")) for cam_cfg in camera_configs(settings)}
        if not camera_id:
            camera_id = default_camera_id(settings)
        if camera_id not in valid_ids:
            raise ValueError(f"Unknown camera id: {camera_id}")
        return camera_id

    def _calibration_camera_config(camera_id: str) -> dict:
        from scanner.calibration import camera_config_by_id

        try:
            return camera_config_by_id(settings, camera_id)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

    def _camera_calib_state(camera_id: str) -> dict:
        sessions = _camera_calib_session.setdefault("sessions", {})
        if camera_id not in sessions:
            sessions[camera_id] = {"images": [], "captures": [], "last_report": None}
        return sessions[camera_id]

    def _extrinsics_turntable_state(camera_id: str) -> dict:
        sessions = _extrinsics_turntable_session.setdefault("sessions", {})
        if camera_id not in sessions:
            sessions[camera_id] = {"captures": [], "last_report": None}
        return sessions[camera_id]

    def _extrinsics_turntable_summary(camera_id: str | None = None) -> dict:
        try:
            selected_id = camera_id or _selected_camera_id()
            cam_cfg = _calibration_camera_config(selected_id)
        except ValueError:
            from scanner.calibration import default_camera_id

            selected_id = default_camera_id(settings)
            cam_cfg = _calibration_camera_config(selected_id)
        with _extrinsics_turntable_lock:
            state = _extrinsics_turntable_state(selected_id)
            captures = [
                {
                    "index": idx + 1,
                    "angle_deg": item["angle_deg"],
                    "quality": item.get("quality", {}),
                    "preview_jpeg_base64": item.get("preview_jpeg_base64"),
                }
                for idx, item in enumerate(state["captures"])
            ]
            report = state.get("last_report")
        return {
            "camera_id": selected_id,
            "camera_label": _camera_label(selected_id),
            "count": len(captures),
            "recommended_min": 6,
            "captures": captures,
            "last_report": report,
            "extrinsics_path": _project_path(cam_cfg.get("extrinsics_path")),
        }

    def _camera_calib_summary(camera_id: str | None = None) -> dict:
        try:
            selected_id = camera_id or _selected_camera_id()
            cam_cfg = _calibration_camera_config(selected_id)
        except ValueError:
            from scanner.calibration import default_camera_id

            selected_id = default_camera_id(settings)
            cam_cfg = _calibration_camera_config(selected_id)

        with _camera_calib_lock:
            state = _camera_calib_state(selected_id)
            captures = list(state["captures"])
            report = state.get("last_report")
        intrinsics_path = _project_path(cam_cfg.get("intrinsics_path"))
        extrinsics_path = _project_path(cam_cfg.get("extrinsics_path"))
        return {
            "camera_id": selected_id,
            "camera_label": _camera_label(selected_id),
            "count": len(captures),
            "recommended_min": 12,
            "recommended_max": 20,
            "captures": captures,
            "last_report": report,
            "intrinsics_path": intrinsics_path,
            "intrinsics_exists": bool(intrinsics_path and os.path.exists(intrinsics_path)),
            "extrinsics_path": extrinsics_path,
            "extrinsics_exists": bool(extrinsics_path and os.path.exists(extrinsics_path)),
        }

    def _laser_calib_summary() -> dict:
        with _laser_calib_lock:
            captures = [
                {
                    "index": idx + 1,
                    "z_mm": item["z_mm"],
                    "previews": item.get("previews", {}),
                    "metrics": item.get("metrics", {}),
                }
                for idx, item in enumerate(_laser_calib_session["captures"])
            ]
            last_result = _laser_calib_session.get("last_result")
        return {
            "count": len(captures),
            "recommended_min": 4,
            "captures": captures,
            "last_result": last_result,
        }

    def _parse_camera_board_payload() -> tuple[str, tuple[int, int], float, bool, int | None, float | None]:
        payload = request.get_json(silent=True) or {}
        source = payload if payload else request.values
        camera_id = str(source.get("camera") or source.get("camera_id") or _selected_camera_id())
        _calibration_camera_config(camera_id)
        board_size = (
            int(source.get("board_cols", 9)),
            int(source.get("board_rows", 6)),
        )
        square_mm = float(source.get("square_size_mm", 25.0))
        auto_bracket = str(source.get("auto_bracket", "false")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        exposure_raw = source.get("calibration_exposure_us", "")
        gain_raw = source.get("calibration_gain", "")
        calibration_exposure_us = int(exposure_raw) if str(exposure_raw).strip() else None
        calibration_gain = float(gain_raw) if str(gain_raw).strip() else None
        return (
            camera_id,
            board_size,
            square_mm,
            auto_bracket,
            calibration_exposure_us,
            calibration_gain,
        )

    def _parse_vector3(source, prefix: str, default: tuple[float, float, float]) -> list[float]:
        values = []
        for axis, fallback in zip(("x", "y", "z"), default):
            raw = source.get(f"{prefix}_{axis}", fallback)
            values.append(float(raw))
        return values

    def _platform_rotation_config() -> tuple[list[float], list[float]]:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "platform.yaml"
        data = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception:
                logger.warning("Could not load platform rotation config", exc_info=True)
        return (
            list(data.get("rotation_axis", [0.0, 1.0, 0.0])),
            list(data.get("rotation_axis_point_mm", [0.0, 0.0, 0.0])),
        )

    def _encode_jpeg_base64(frame, quality: int = 85) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Could not encode JPEG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _camera_label(camera_id: str) -> str:
        from scanner.calibration import camera_configs

        for cam_cfg in camera_configs(settings):
            if str(cam_cfg.get("id")) != str(camera_id):
                continue
            display_name = cam_cfg.get("display_name")
            if display_name:
                return str(display_name)
            cam_type = str(cam_cfg.get("type", "")).lower()
            if cam_type == "usb":
                return "USB"
            if cam_type in ("pi", "picamera", "csi"):
                return "Nape"
        return str(camera_id).upper()

    def _camera_view_configs() -> list[dict]:
        from scanner.calibration import camera_configs

        return [
            {
                "id": str(cam_cfg.get("id", "main")),
                "label": _camera_label(str(cam_cfg.get("id", "main"))),
                "type": str(cam_cfg.get("type", "pi")),
                "resolution": cam_cfg.get("resolution", settings.get("camera", {}).get("resolution", [640, 480])),
                "exposure_us": int(cam_cfg.get("exposure_us", settings.get("camera", {}).get("exposure_us", 1000))),
                "gain": float(cam_cfg.get("gain", settings.get("camera", {}).get("gain", 1.0))),
                "calibration_exposure_us": int(
                    cam_cfg.get(
                        "calibration_exposure_us",
                        max(
                            5000,
                            int(
                                cam_cfg.get(
                                    "exposure_us",
                                    settings.get("camera", {}).get("exposure_us", 1000),
                                )
                            )
                            * 6,
                        ),
                    )
                ),
                "calibration_gain": float(
                    cam_cfg.get(
                        "calibration_gain",
                        cam_cfg.get("gain", settings.get("camera", {}).get("gain", 1.0)),
                    )
                ),
                "lens_position": cam_cfg.get("lens_position"),
                "focus_absolute": cam_cfg.get("focus_absolute"),
                "pixel_format": cam_cfg.get("pixel_format", ""),
                "device_path": cam_cfg.get("device_path"),
                "laser_threshold": int(
                    cam_cfg.get(
                        "laser_threshold",
                        settings.get("processing", {}).get("laser_threshold", 180),
                    )
                ),
                "laser_mask": cam_cfg.get("laser_mask", []) or [],
                "laser_sampling": cam_cfg.get("laser_sampling", {}) or {},
            }
            for cam_cfg in camera_configs(settings)
        ]

    def _parse_mask_rects(raw) -> list:
        if raw in (None, "", []):
            return []
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict) and raw.get("enabled"):
            raw = [[raw.get("x0", 0), raw.get("y0", 0), raw.get("x1", 0), raw.get("y1", 0)]]
        if not isinstance(raw, list):
            return []

        masks = []
        for item in raw:
            if isinstance(item, dict):
                if item.get("points"):
                    item = item.get("points")
                else:
                    item = [
                        item.get("x0", 0),
                        item.get("y0", 0),
                        item.get("x1", 0),
                        item.get("y1", 0),
                    ]
            if not isinstance(item, list | tuple):
                continue
            if len(item) == 4 and all(isinstance(value, int | float | str) for value in item):
                masks.append([int(float(value)) for value in item])
                continue

            points = []
            for point in item:
                if not isinstance(point, list | tuple) or len(point) != 2:
                    points = []
                    break
                points.append([int(float(point[0])), int(float(point[1]))])
            if len(points) >= 3:
                masks.append(points)
        return masks

    def _sampling_config(cam_cfg: dict | None = None) -> dict[str, int]:
        sampling = (cam_cfg or {}).get("laser_sampling", {}) or {}
        return {
            "x_stride": max(1, int(sampling.get("x_stride", 1))),
            "y_stride": max(1, int(sampling.get("y_stride", 1))),
            "x_offset": max(0, int(sampling.get("x_offset", 0))),
            "y_offset": max(0, int(sampling.get("y_offset", 0))),
        }

    def _camera_processing_config(
        camera_id: str | None = None,
    ) -> tuple[int, list, dict[str, int]]:
        proc_cfg = settings.get("processing", {})
        threshold = int(proc_cfg.get("laser_threshold", 180))
        mask = []
        sampling = _sampling_config()
        if camera_id:
            from scanner.calibration import camera_configs

            for cam_cfg in camera_configs(settings):
                if str(cam_cfg.get("id")) == str(camera_id):
                    threshold = int(cam_cfg.get("laser_threshold", threshold))
                    mask = cam_cfg.get("laser_mask", []) or []
                    sampling = _sampling_config(cam_cfg)
                    break
        return threshold, mask, sampling

    def _artifact_tabs() -> list[dict]:
        tabs: list[dict] = []
        for cam in _camera_view_configs():
            tabs.append(
                {
                    "kind": f"extract_{cam['id']}",
                    "label": f"Extraction {cam['label']}",
                }
            )
        for cam in _camera_view_configs():
            tabs.append(
                {
                    "kind": f"cloud_{cam['id']}",
                    "label": f"Nuage {cam['label']}",
                }
            )
        tabs.extend(
            [
                {"kind": "cloud_combined", "label": "Nuage combine"},
                {"kind": "mesh", "label": "STL"},
            ]
        )
        return tabs

    def _artifact_public(item: dict) -> dict:
        path = item.get("path")
        result = dict(item)
        result["available"] = bool(path and os.path.exists(path))
        result.pop("path", None)
        return result

    def _initial_artifacts() -> dict:
        artifacts: dict = {}
        for cam_cfg in settings.get("cameras", []):
            camera_id = str(cam_cfg.get("id", "main"))
            label = _camera_label(camera_id)
            artifacts[f"extract_{camera_id}"] = {
                "kind": f"extract_{camera_id}",
                "path": os.path.join("/tmp/scan_frames", f"latest_{camera_id}.jpg"),
                "label": f"Extraction {label}",
                "media_type": "image/jpeg",
                "stage": "extraction",
            }
            artifacts[f"cloud_{camera_id}"] = {
                "kind": f"cloud_{camera_id}",
                "path": None,
                "label": f"Nuage {label}",
                "media_type": "model/ply",
                "stage": "cloud",
            }
        artifacts["cloud_combined"] = {
            "kind": "cloud_combined",
            "path": None,
            "label": "Nuage combine",
            "media_type": "model/ply",
            "stage": "cloud",
        }
        artifacts["mesh"] = {
            "kind": "mesh",
            "path": None,
            "label": "STL final",
            "media_type": "model/stl",
            "stage": "mesh",
        }
        return artifacts

    def _public_artifacts() -> dict:
        with _scan_lock:
            artifacts = dict(_scan_state.get("artifacts", {}))
        return {key: _artifact_public(value) for key, value in artifacts.items()}

    def _capture_checkerboard_candidate(
        camera_id: str,
        board_size: tuple[int, int],
        auto_bracket: bool,
        calibration_exposure_us: int | None,
        calibration_gain: float | None,
    ) -> tuple:
        from scanner.calibration import checkerboard_capture_quality, draw_checkerboard_overlay
        from scanner.hardware import HardwareError, camera_capture, camera_set_exposure, laser_set

        cam_cfg = _calibration_camera_config(camera_id)
        scan_exposure = int(cam_cfg.get("exposure_us", settings.get("camera", {}).get("exposure_us", 1000)))
        scan_gain = float(cam_cfg.get("gain", settings.get("camera", {}).get("gain", 1.0)))
        base_exposure = int(
            calibration_exposure_us
            if calibration_exposure_us is not None
            else cam_cfg.get("calibration_exposure_us", max(5000, scan_exposure * 6))
        )
        gain = float(
            calibration_gain
            if calibration_gain is not None
            else cam_cfg.get("calibration_gain", scan_gain)
        )
        exposures = [base_exposure]
        exposure_note = None
        if auto_bracket:
            exposures = [max(100, int(base_exposure * factor)) for factor in (0.5, 1.0, 2.0, 4.0)]

        with _camera_calib_lock:
            state = _camera_calib_state(camera_id)
            previous_poses = [c["pose"] for c in state["captures"] if c.get("pose")]

        best = None
        try:
            laser_set(False)
            for exposure in exposures:
                try:
                    camera_set_exposure(exposure, gain, camera_id=camera_id)
                    time.sleep(0.08)
                except HardwareError as exc:
                    exposure_note = f"reglage exposition indisponible: {exc}"
                    if auto_bracket:
                        logger.warning("Calibration exposure bracketing unavailable: %s", exc)
                    exposures = [base_exposure]
                frame = camera_capture(camera_id)
                quality = checkerboard_capture_quality(frame, board_size, previous_poses=previous_poses)
                candidate = (quality.get("score", 0.0), exposure, frame, quality)
                if best is None or candidate[0] > best[0]:
                    best = candidate
                if quality.get("accepted") or exposure_note:
                    break
        finally:
            try:
                laser_set(False)
            except Exception:
                pass
            try:
                camera_set_exposure(scan_exposure, scan_gain, camera_id=camera_id)
            except Exception:
                pass

        if best is None:
            raise HardwareError("Camera capture failed")
        _score, exposure, frame, quality = best
        overlay = draw_checkerboard_overlay(frame, board_size, quality)
        quality = dict(quality)
        quality.pop("corners", None)
        quality["exposure_us"] = exposure
        if exposure_note:
            quality["exposure_note"] = exposure_note
        return frame, overlay, quality

    def _push_sse(data: dict) -> None:
        """Push a dict as an SSE event."""
        try:
            _sse_queue.put_nowait(data)
        except queue.Full:
            pass  # Drop oldest — UI will re-poll

    def _on_state_transition(
        old_state: ScannerState, new_state: ScannerState
    ) -> None:
        with _scan_lock:
            _scan_state["state"] = new_state.name
        _push_sse({"state": new_state.name, "progress": _scan_state["progress"]})
        from scanner.interface.display_local import update_display
        update_display(new_state.name, _scan_state["progress"])

    _sm.add_observer(_on_state_transition)

    # ------------------------------------------------------------------ #
    # Routes — Main page
    # ------------------------------------------------------------------ #

    @app.route("/")
    def index() -> str:
        with _scan_lock:
            state = dict(_scan_state)
            state["artifacts"] = {
                key: _artifact_public(value)
                for key, value in _scan_state.get("artifacts", {}).items()
            }
        return render_template(
            "index.html",
            scan_state=state,
            artifact_tabs=_artifact_tabs(),
        )

    # ------------------------------------------------------------------ #
    # Routes — Scan control
    # ------------------------------------------------------------------ #

    @app.route("/scan/start", methods=["POST"])
    def scan_start() -> Response:
        """Start a scan in a background thread."""
        with _scan_lock:
            if _scan_state["state"] not in ("IDLE", "COMPLETE", "ERROR"):
                return jsonify({"error": "Scan already in progress"}), 409
            # COMPLETE/ERROR → SCANNING is invalid; reset to IDLE first
            if _sm.current_state.name in ("COMPLETE", "ERROR"):
                _sm.reset()
            _scan_state["state"] = "IDLE"
            _scan_state["progress"] = 0
            _scan_state["error"] = None
            _scan_state["artifacts"] = _initial_artifacts()

        def _run() -> None:
            from scanner.orchestration.scan import run_scan

            def _cb(current: int, total: int, message: str) -> None:
                pct = int(100 * current / max(total, 1))
                with _scan_lock:
                    _scan_state["progress"] = pct
                    _scan_state["message"] = message
                _push_sse({"state": _scan_state["state"], "progress": pct, "message": message})
                from scanner.interface.display_local import update_display
                update_display(_scan_state["state"], pct)

            def _artifact_cb(artifact: dict) -> None:
                kind = str(artifact.get("kind", ""))
                if not kind:
                    return
                with _scan_lock:
                    current = dict(_scan_state.get("artifacts", {}).get(kind, {}))
                    current.update(artifact)
                    if kind.startswith("extract_"):
                        camera_id = kind.removeprefix("extract_")
                        current["label"] = f"Extraction {_camera_label(camera_id)}"
                        current["stage"] = "extraction"
                    elif kind.startswith("cloud_") and kind not in ("cloud_combined",):
                        camera_id = kind.removeprefix("cloud_")
                        current["label"] = f"Nuage {_camera_label(camera_id)}"
                        current["stage"] = "cloud"
                    elif kind == "cloud_combined":
                        current["stage"] = "cloud"
                    elif kind == "mesh":
                        current["stage"] = "mesh"
                    _scan_state.setdefault("artifacts", {})[kind] = current
                    public = _artifact_public(current)
                _push_sse({"artifact": public, "artifacts": _public_artifacts()})

            try:
                path = run_scan(
                    settings,
                    progress_callback=_cb,
                    artifact_callback=_artifact_cb,
                    state_machine=_sm,
                )
                with _scan_lock:
                    _scan_state["last_file"] = path
                    _scan_state["error"] = None
            except Exception as exc:
                logger.error("Scan thread error: %s", exc)
                with _scan_lock:
                    _scan_state["error"] = str(exc)

        thread = threading.Thread(target=_run, daemon=True, name="scan-worker")
        thread.start()
        return jsonify({"status": "started"}), 202

    @app.route("/scan/status")
    def scan_status() -> Response:
        """Return current scan state as JSON."""
        with _scan_lock:
            state = dict(_scan_state)
            state["artifacts"] = {
                key: _artifact_public(value)
                for key, value in _scan_state.get("artifacts", {}).items()
            }
        return jsonify(state)

    @app.route("/scan/artifacts")
    def scan_artifacts() -> Response:
        """Return scan intermediate artifacts and availability."""
        return jsonify(_public_artifacts())

    @app.route("/scan/artifact/<kind>")
    def scan_artifact(kind: str) -> Response:
        """Serve one scan artifact by stable key."""
        with _scan_lock:
            artifact = dict(_scan_state.get("artifacts", {}).get(kind, {}))
        path = artifact.get("path")
        if not path or not os.path.exists(path):
            return jsonify({"error": f"Artifact unavailable: {kind}"}), 404
        media_type = str(artifact.get("media_type") or "application/octet-stream")
        return send_file(
            path,
            mimetype=media_type,
            as_attachment=False,
            download_name=os.path.basename(path),
        )

    @app.route("/scan/download")
    def scan_download() -> Response:
        """Download the last exported mesh file."""
        with _scan_lock:
            path = _scan_state.get("last_file")
        if not path or not os.path.exists(path):
            return jsonify({"error": "No scan file available"}), 404
        return send_file(
            path,
            as_attachment=True,
            download_name=os.path.basename(path),
        )

    @app.route("/scan/frame/latest")
    def scan_frame_latest() -> Response:
        """Return the most recently saved scan frame as JPEG."""
        camera_id = request.args.get("camera")
        name = "latest.jpg" if not camera_id else f"latest_{camera_id}.jpg"
        path = os.path.join("/tmp/scan_frames", name)
        if not os.path.exists(path):
            return Response(status=404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/scan/frame/<int:step>")
    def scan_frame_step(step: int) -> Response:
        """Return the JPEG frame captured at the given step index."""
        camera_id = request.args.get("camera")
        suffix = "" if not camera_id else f"_{camera_id}"
        path = f"/tmp/scan_frames/frame_{step:03d}{suffix}.jpg"
        if not os.path.exists(path):
            return Response(status=404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/scan/stream")
    def scan_stream() -> Response:
        """Server-Sent Events stream for real-time scan progress."""

        def _generate():
            yield "retry: 2000\n\n"
            while True:
                try:
                    data = _sse_queue.get(timeout=15)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"

        return Response(
            stream_with_context(_generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------ #
    # Routes — Calibration
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Routes — Preview (mock camera visualisation)
    # ------------------------------------------------------------------ #

    @app.route("/preview")
    def preview_page() -> str:
        n_steps = settings.get("scan", {}).get("n_steps", 200)
        return render_template("preview.html", n_steps=n_steps)

    @app.route("/preview/frame")
    def preview_frame() -> Response:
        """Return a mock camera frame as JPEG for the given rotation angle."""
        import cv2
        from scanner.hardware.mock import MockCamera

        angle_rad = float(request.args.get("angle", 0.0))
        cam_cfg = settings.get("camera", {"resolution": [640, 480]})
        cam = MockCamera(cam_cfg)
        cam.set_rotation_angle(angle_rad)
        frame = cam.capture()

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    @app.route("/preview/extraction")
    def preview_extraction() -> Response:
        """Return a frame with laser line detection overlay as JPEG."""
        import cv2
        from scanner.hardware.mock import MockCamera
        from scanner.processing import extract_laser_line

        angle_rad = float(request.args.get("angle", 0.0))
        cam_cfg = settings.get("camera", {"resolution": [640, 480]})
        cam = MockCamera(cam_cfg)
        cam.set_rotation_angle(angle_rad)
        frame = cam.capture()

        proc_cfg = settings.get("processing", {})
        threshold, mask_rects, sampling = _camera_processing_config()
        min_px = int(proc_cfg.get("min_line_pixels", 10))
        subpixel = bool(proc_cfg.get("subpixel", True))
        extraction_mode = str(proc_cfg.get("extraction_mode", "row_mean"))

        line = extract_laser_line(
            frame,
            threshold=threshold,
            min_pixels=min_px,
            subpixel=subpixel,
            mode=extraction_mode,
            mask_rects=mask_rects,
            **sampling,
        )
        # Draw detected pixels as red dots on the frame
        overlay = frame.copy()
        for i in range(line.shape[0]):
            col, row = int(round(line[i, 0])), int(round(line[i, 1]))
            cv2.circle(overlay, (col, row), 1, (0, 0, 255), -1)

        _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    # ------------------------------------------------------------------ #
    # Routes — Manual hardware control
    # ------------------------------------------------------------------ #

    def _manual_allowed() -> bool:
        """Manual controls are disabled while a scan is in progress."""
        with _scan_lock:
            state = _scan_state.get("state", "IDLE")
        return state in ("IDLE", "COMPLETE", "ERROR")

    @app.route("/manual")
    def manual_page() -> str:
        with _scan_lock:
            state = dict(_scan_state)
        return render_template(
            "manual.html",
            scan_state=state,
            cameras=_camera_view_configs(),
        )

    @app.route("/manual/status")
    def manual_status() -> Response:
        with _scan_lock:
            state = dict(_scan_state)
        return jsonify(
            {
                "scan_state": state.get("state", "IDLE"),
                "manual_allowed": _manual_allowed(),
            }
        )

    @app.route("/camera-config")
    def camera_config_page() -> str:
        with _scan_lock:
            state = dict(_scan_state)
        return render_template(
            "camera_config.html",
            scan_state=state,
            cameras=_camera_view_configs(),
        )

    @app.route("/camera-config/status")
    def camera_config_status() -> Response:
        from scanner.hardware import HardwareError, camera_get_info

        result = {}
        for cam in _camera_view_configs():
            camera_id = cam["id"]
            try:
                result[camera_id] = {
                    "label": cam["label"],
                    "type": cam["type"],
                    "info": camera_get_info(camera_id),
                }
            except HardwareError as exc:
                result[camera_id] = {"label": cam["label"], "type": cam["type"], "error": str(exc)}
        return jsonify(result)

    @app.route("/camera-config/apply", methods=["POST"])
    def camera_config_apply() -> Response:
        from scanner.hardware import HardwareError, camera_set_controls

        if not _manual_allowed():
            return jsonify({"error": "Camera config disabled while scan is running"}), 409

        data = request.get_json(silent=True) or {}
        camera_id = str(data.get("camera", ""))
        if not camera_id:
            return jsonify({"error": "camera is required"}), 400

        controls = {}
        for key in ("width", "height", "exposure_us"):
            if data.get(key) not in (None, ""):
                controls[key] = int(data[key])
        for key in ("gain", "lens_position"):
            if data.get(key) not in (None, ""):
                controls[key] = float(data[key])
        if data.get("focus_absolute") not in (None, ""):
            controls["focus_absolute"] = int(float(data["focus_absolute"]))
        if data.get("pixel_format"):
            controls["pixel_format"] = str(data["pixel_format"]).strip()

        try:
            return jsonify({"status": "ok", "camera": camera_id, "info": camera_set_controls(camera_id, controls)})
        except (HardwareError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/camera-config/formats")
    def camera_config_formats() -> Response:
        import shutil
        import subprocess

        camera_id = str(request.args.get("camera", ""))
        cam = next((item for item in _camera_view_configs() if item["id"] == camera_id), None)
        if not cam:
            return jsonify({"error": f"Unknown camera: {camera_id}"}), 404
        device_path = cam.get("device_path")
        if not device_path:
            return jsonify({"error": "No V4L2 device path configured for this camera"}), 400
        if not shutil.which("v4l2-ctl"):
            return jsonify({"error": "v4l2-ctl is not installed"}), 400
        try:
            proc = subprocess.run(
                ["v4l2-ctl", "--list-formats-ext", "-d", str(device_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            return jsonify({"camera": camera_id, "device_path": device_path, "formats": proc.stdout})
        except subprocess.CalledProcessError as exc:
            return jsonify({"error": exc.stderr or str(exc)}), 500
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/manual/camera/frame")
    def manual_camera_frame() -> Response:
        """Capture and return one live camera JPEG frame for manual mode."""
        import cv2
        from scanner.hardware import HardwareError, camera_capture

        if not _manual_allowed():
            return Response(status=409)
        try:
            frame = camera_capture(request.args.get("camera"))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return Response(status=500)
            return Response(buf.tobytes(), mimetype="image/jpeg")
        except HardwareError:
            return Response(status=503)

    @app.route("/manual/camera/tuning")
    def manual_camera_tuning() -> Response:
        """Capture a frame and overlay detected laser pixels in red.

        Query params:
            threshold: green-dominant laser threshold (default from settings)
            min_pixels: minimum columns to validate a line (default from settings)
        """
        import cv2
        import numpy as np
        from scanner.hardware import HardwareError, camera_capture
        from scanner.processing import extract_laser_line

        if not _manual_allowed():
            return Response(status=409)

        proc_cfg = settings.get("processing", {})
        camera_id = request.args.get("camera")
        default_threshold, mask_rects, sampling = _camera_processing_config(camera_id)
        default_min_px = int(proc_cfg.get("min_line_pixels", 15))
        extraction_mode = str(proc_cfg.get("extraction_mode", "row_mean"))

        try:
            threshold = int(request.args.get("threshold", default_threshold))
        except ValueError:
            threshold = default_threshold
        try:
            min_px = int(request.args.get("min_pixels", default_min_px))
        except ValueError:
            min_px = default_min_px
        threshold = max(0, min(255, threshold))
        min_px = max(1, min(640, min_px))
        if "mask" in request.args:
            try:
                mask_rects = _parse_mask_rects(request.args.get("mask"))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        try:
            frame = camera_capture(camera_id)
        except HardwareError:
            return Response(status=503)

        line = extract_laser_line(
            frame,
            threshold=threshold,
            min_pixels=min_px,
            subpixel=True,
            mode=extraction_mode,
            camera_id=camera_id,
            mask_rects=mask_rects,
            **sampling,
        )

        signal = frame[:, :, 1]  # green channel only
        gr_max = int(signal.max())
        gr_mean = float(signal.mean())

        overlay = frame.copy()
        for mask in mask_rects:
            if len(mask) == 4 and all(isinstance(value, int | float) for value in mask):
                x0, y0, x1, y1 = [int(v) for v in mask]
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
                continue
            try:
                pts = np.asarray(mask, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
            except Exception:
                continue
        for i in range(line.shape[0]):
            col, row = int(round(line[i, 0])), int(round(line[i, 1]))
            cv2.circle(overlay, (col, row), 2, (0, 0, 255), -1)

        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return Response(status=500)

        resp = Response(buf.tobytes(), mimetype="image/jpeg")
        resp.headers["X-Detected-Columns"] = str(line.shape[0])
        resp.headers["X-GR-Max"] = str(gr_max)
        resp.headers["X-GR-Mean"] = f"{gr_mean:.2f}"
        resp.headers["X-Threshold"] = str(threshold)
        resp.headers["X-Min-Pixels"] = str(min_px)
        resp.headers["X-Mask-Rects"] = str(len(mask_rects))
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Detected-Columns,X-GR-Max,X-GR-Mean,X-Threshold,X-Min-Pixels,X-Mask-Rects"
        )
        return resp

    @app.route("/manual/processing/apply", methods=["POST"])
    def manual_processing_apply() -> Response:
        if not _manual_allowed():
            return jsonify({"error": "Manual control disabled while scan is running"}), 409

        data = request.get_json(silent=True) or {}
        camera_id = str(data.get("camera", ""))
        if not camera_id:
            return jsonify({"error": "camera is required"}), 400

        try:
            threshold = max(0, min(255, int(data.get("threshold"))))
            mask_rects = _parse_mask_rects(data.get("mask"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Invalid processing config: {exc}"}), 400

        for cam_cfg in settings.get("cameras", []):
            if str(cam_cfg.get("id")) == camera_id:
                cam_cfg["laser_threshold"] = threshold
                cam_cfg["laser_mask"] = mask_rects
                return jsonify(
                    {
                        "status": "ok",
                        "camera": camera_id,
                        "laser_threshold": threshold,
                        "laser_mask": mask_rects,
                    }
                )
        return jsonify({"error": f"Unknown camera: {camera_id}"}), 404

    @app.route("/manual/laser", methods=["POST"])
    def manual_laser() -> Response:
        from scanner.hardware import HardwareError, laser_set

        if not _manual_allowed():
            return jsonify({"error": "Manual control disabled while scan is running"}), 409

        data = request.get_json(silent=True) or {}
        state = bool(data.get("state", False))

        try:
            laser_set(state)
            return jsonify({"status": "ok", "laser": "on" if state else "off"})
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/manual/motor", methods=["POST"])
    def manual_motor() -> Response:
        from scanner.hardware import HardwareError, motor_step

        if not _manual_allowed():
            return jsonify({"error": "Manual control disabled while scan is running"}), 409

        data = request.get_json(silent=True) or {}
        try:
            steps = int(data.get("steps", 1))
            direction = str(data.get("direction", "clockwise"))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid payload"}), 400

        if steps < 1 or steps > 5000:
            return jsonify({"error": "steps must be between 1 and 5000"}), 400
        if direction not in ("clockwise", "counterclockwise"):
            return jsonify({"error": "direction must be clockwise or counterclockwise"}), 400

        try:
            motor_step(steps, direction=direction)
            return jsonify({"status": "ok", "steps": steps, "direction": direction})
        except (HardwareError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/manual/led", methods=["POST"])
    def manual_led() -> Response:
        from scanner.hardware import HardwareError, led_blink, led_set

        if not _manual_allowed():
            return jsonify({"error": "Manual control disabled while scan is running"}), 409

        data = request.get_json(silent=True) or {}
        try:
            color = str(data.get("color", "orange")).lower()
            mode = str(data.get("mode", "off")).lower()
            frequency_hz = float(data.get("frequency_hz", 1.0))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid payload"}), 400

        if color not in ("orange", "red"):
            return jsonify({"error": "color must be orange or red"}), 400
        if mode not in ("on", "off", "blink"):
            return jsonify({"error": "mode must be on, off or blink"}), 400
        if frequency_hz <= 0:
            return jsonify({"error": "frequency_hz must be > 0"}), 400

        try:
            if mode == "on":
                led_set(color, True)
            elif mode == "off":
                led_set(color, False)
            else:
                led_blink(color, frequency_hz)
            return jsonify(
                {
                    "status": "ok",
                    "color": color,
                    "mode": mode,
                    "frequency_hz": frequency_hz if mode == "blink" else None,
                }
            )
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/manual/safe-off", methods=["POST"])
    def manual_safe_off() -> Response:
        from scanner.hardware import HardwareError, laser_set, led_set

        if not _manual_allowed():
            return jsonify({"error": "Manual control disabled while scan is running"}), 409

        try:
            laser_set(False)
            for color in ("orange", "red"):
                led_set(color, False)
            return jsonify({"status": "ok"})
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/calibration")
    def calibration_page() -> str:
        use_checkerboard = bool(settings.get("calibration", {}).get("use_checkerboard", True))
        camera_views = _camera_view_configs()
        selected_camera = str(request.args.get("camera") or camera_views[0]["id"])
        return render_template(
            "calibration.html",
            use_checkerboard=use_checkerboard,
            cameras=camera_views,
            selected_camera=selected_camera,
            camera_calib=_camera_calib_summary(selected_camera),
        )

    @app.route("/calibration/camera/session")
    def calibration_camera_session() -> Response:
        """Return current in-memory checkerboard capture session."""
        try:
            camera_id = _selected_camera_id()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(_camera_calib_summary(camera_id))

    @app.route("/calibration/camera/reset", methods=["POST"])
    def calibration_camera_reset() -> Response:
        """Clear captured checkerboard images."""
        try:
            camera_id = _selected_camera_id()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _camera_calib_lock:
            state = _camera_calib_state(camera_id)
            state["images"] = []
            state["captures"] = []
            state["last_report"] = None
        return jsonify(_camera_calib_summary(camera_id))

    @app.route("/calibration/camera/frame")
    def calibration_camera_frame() -> Response:
        """Return one live checkerboard frame with detection overlay."""
        import cv2
        from scanner.calibration import checkerboard_capture_quality, draw_checkerboard_overlay
        from scanner.hardware import HardwareError, camera_capture, laser_set

        if not _manual_allowed():
            return Response(status=409)
        try:
            (
                camera_id,
                board_size,
                _square_mm,
                _auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            ) = _parse_camera_board_payload()
        except (TypeError, ValueError):
            from scanner.calibration import default_camera_id

            camera_id = default_camera_id(settings)
            board_size = (9, 6)
            calibration_exposure_us = None
            calibration_gain = None

        try:
            laser_set(False)
            try:
                from scanner.hardware import camera_set_exposure

                cam_cfg = _calibration_camera_config(camera_id)
                scan_exposure = int(
                    cam_cfg.get("exposure_us", settings.get("camera", {}).get("exposure_us", 1000))
                )
                scan_gain = float(
                    cam_cfg.get("gain", settings.get("camera", {}).get("gain", 1.0))
                )
                exposure = int(
                    calibration_exposure_us
                    if calibration_exposure_us is not None
                    else cam_cfg.get("calibration_exposure_us", max(5000, scan_exposure * 6))
                )
                gain = float(
                    calibration_gain
                    if calibration_gain is not None
                    else cam_cfg.get("calibration_gain", scan_gain)
                )
                camera_set_exposure(
                    exposure,
                    gain,
                    camera_id=camera_id,
                )
                time.sleep(0.08)
            except Exception:
                pass
            frame = camera_capture(camera_id)
        except HardwareError:
            return Response(status=503)
        finally:
            try:
                laser_set(False)
            except Exception:
                pass
            try:
                from scanner.hardware import camera_set_exposure

                cam_cfg = _calibration_camera_config(camera_id)
                camera_set_exposure(
                    int(
                        cam_cfg.get(
                            "exposure_us",
                            settings.get("camera", {}).get("exposure_us", 1000),
                        )
                    ),
                    float(cam_cfg.get("gain", settings.get("camera", {}).get("gain", 1.0))),
                    camera_id=camera_id,
                )
            except Exception:
                pass

        with _camera_calib_lock:
            state = _camera_calib_state(camera_id)
            previous_poses = [c["pose"] for c in state["captures"] if c.get("pose")]
        quality = checkerboard_capture_quality(frame, board_size, previous_poses=previous_poses)
        overlay = draw_checkerboard_overlay(frame, board_size, quality)
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return Response(status=500)

        metrics = quality.get("metrics", {})
        resp = Response(buf.tobytes(), mimetype="image/jpeg")
        resp.headers["X-Checkerboard-Found"] = str(bool(quality.get("found"))).lower()
        resp.headers["X-Checkerboard-Accepted"] = str(bool(quality.get("accepted"))).lower()
        resp.headers["X-Checkerboard-Status"] = str(quality.get("status", ""))
        resp.headers["X-Brightness-Mean"] = f"{metrics.get('brightness_mean', 0.0):.2f}"
        resp.headers["X-Saturated-Pct"] = f"{metrics.get('saturated_pct', 0.0):.2f}"
        resp.headers["X-Contrast-Std"] = f"{metrics.get('contrast_std', 0.0):.2f}"
        resp.headers["X-Sharpness"] = f"{metrics.get('sharpness', 0.0):.2f}"
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Checkerboard-Found,X-Checkerboard-Accepted,X-Checkerboard-Status,"
            "X-Brightness-Mean,X-Saturated-Pct,X-Contrast-Std,X-Sharpness"
        )
        return resp

    @app.route("/calibration/camera/capture", methods=["POST"])
    def calibration_camera_capture() -> Response:
        """Capture one checkerboard image and keep it if quality is acceptable."""
        from scanner.hardware import HardwareError

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409
        try:
            (
                camera_id,
                board_size,
                _square_mm,
                auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            ) = _parse_camera_board_payload()
            frame, overlay, quality = _capture_checkerboard_candidate(
                camera_id,
                board_size,
                auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid checkerboard payload: {exc}"}), 400
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            logger.error("Checkerboard capture error: %s", exc)
            return jsonify({"error": "Internal error during checkerboard capture"}), 500

        preview = _encode_jpeg_base64(overlay)
        accepted = bool(quality.get("accepted"))
        if accepted:
            metrics = quality.get("metrics", {})
            with _camera_calib_lock:
                state = _camera_calib_state(camera_id)
                idx = len(state["images"]) + 1
                state["images"].append(frame)
                state["captures"].append(
                    {
                        "index": idx,
                        "pose": quality.get("pose"),
                        "metrics": metrics,
                        "exposure_us": quality.get("exposure_us"),
                        "preview_jpeg_base64": preview,
                    }
                )

        result = _camera_calib_summary(camera_id)
        result.update(
            {
                "accepted": accepted,
                "quality": quality,
                "preview_jpeg_base64": preview,
            }
        )
        return jsonify(result), 200 if accepted else 422

    @app.route("/calibration/camera/run", methods=["POST"])
    def calibration_camera_run() -> Response:
        """Run calibration from the in-memory guided capture session."""
        from scanner.calibration import CalibrationError, calibrate_camera_with_report

        try:
            (
                camera_id,
                board_size,
                square_mm,
                _auto_bracket,
                _calibration_exposure_us,
                _calibration_gain,
            ) = _parse_camera_board_payload()
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid checkerboard payload: {exc}"}), 400

        with _camera_calib_lock:
            state = _camera_calib_state(camera_id)
            images = list(state["images"])
        if len(images) < 4:
            return jsonify({"error": f"At least 4 accepted images are required, got {len(images)}"}), 422

        try:
            cam_cfg = _calibration_camera_config(camera_id)
            output_path = _project_path(cam_cfg.get("intrinsics_path"))
            camera_matrix, dist_coeffs, report = calibrate_camera_with_report(
                images,
                board_size=board_size,
                square_size_mm=square_mm,
                output_path=output_path,
            )
            result = {
                "status": "ok",
                "camera": camera_id,
                "output_path": output_path,
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "dist_coeffs": dist_coeffs.tolist(),
                "report": report,
            }
            with _camera_calib_lock:
                state = _camera_calib_state(camera_id)
                state["last_report"] = report
            return jsonify(result)
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.exception("Guided camera calibration error")
            return jsonify({"error": f"Internal error during calibration: {exc}"}), 500

    @app.route("/calibration/extrinsics/run", methods=["POST"])
    def calibration_extrinsics_run() -> Response:
        """Capture one known-pose checkerboard image and solve camera extrinsics."""
        from scanner.calibration import (
            CalibrationError,
            calibrate_camera_extrinsics,
            default_extrinsics_path,
            load_camera_calibration,
        )
        from scanner.hardware import HardwareError

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        payload = request.get_json(silent=True) or {}
        source = payload if payload else request.values
        try:
            (
                camera_id,
                board_size,
                square_mm,
                auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            ) = _parse_camera_board_payload()
            board_origin = _parse_vector3(source, "board_origin", (-100.0, 125.0, 0.0))
            board_col_axis = _parse_vector3(source, "board_col_axis", (1.0, 0.0, 0.0))
            board_row_axis = _parse_vector3(source, "board_row_axis", (0.0, -1.0, 0.0))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid extrinsics payload: {exc}"}), 400

        try:
            cam_cfg = _calibration_camera_config(camera_id)
            intrinsics_path = _project_path(cam_cfg.get("intrinsics_path"))
            if not intrinsics_path or not os.path.exists(intrinsics_path):
                return jsonify({"error": f"Camera {camera_id}: intrinsics file missing"}), 409
            camera_matrix, dist_coeffs = load_camera_calibration(intrinsics_path)
            frame, overlay, quality = _capture_checkerboard_candidate(
                camera_id,
                board_size,
                auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            )
            output_path = _project_path(cam_cfg.get("extrinsics_path")) or default_extrinsics_path(camera_id)
            initial_rotation = None
            initial_translation = None
            try:
                from scanner.calibration.multi_camera import _look_at_extrinsics

                if isinstance(cam_cfg.get("extrinsics"), dict) and "position_mm" in cam_cfg["extrinsics"]:
                    initial_rotation, initial_translation = _look_at_extrinsics(cam_cfg["extrinsics"])
            except Exception:
                logger.warning(
                    "Camera %s extrinsics mechanical prior is invalid; solving without prior",
                    camera_id,
                    exc_info=True,
                )
            rotation, translation, report = calibrate_camera_extrinsics(
                frame,
                camera_matrix,
                dist_coeffs,
                board_size,
                square_mm,
                board_origin,
                board_col_axis,
                board_row_axis,
                output_path=output_path,
                initial_camera_to_platform_rotation=initial_rotation,
                initial_camera_to_platform_translation=initial_translation,
                max_prior_distance_mm=float(cam_cfg.get("extrinsics_prior_max_distance_mm", 120.0)),
                max_translation_fallback_distance_mm=float(
                    cam_cfg.get("extrinsics_translation_fallback_max_distance_mm", 250.0)
                ),
                max_fixed_translation_reprojection_px=float(
                    cam_cfg.get("extrinsics_fixed_translation_max_reprojection_px", 5.0)
                ),
            )
            return jsonify(
                {
                    "status": "ok",
                    "camera": camera_id,
                    "output_path": output_path,
                    "intrinsics_path": intrinsics_path,
                    "rotation_matrix": rotation.tolist(),
                    "translation_mm": translation.tolist(),
                    "report": report,
                    "quality": quality,
                    "preview_jpeg_base64": _encode_jpeg_base64(overlay),
                }
            )
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 503
        except CalibrationError as exc:
            payload = {"error": str(exc)}
            report = getattr(exc, "report", None)
            if report is not None:
                payload["report"] = report
            return jsonify(payload), 422
        except Exception as exc:
            logger.exception("Extrinsics calibration error")
            return jsonify({"error": f"Internal error during extrinsics calibration: {exc}"}), 500

    @app.route("/calibration/extrinsics/turntable/session")
    def calibration_extrinsics_turntable_session() -> Response:
        """Return guided rotating-checkerboard extrinsics session."""
        try:
            camera_id = _selected_camera_id()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(_extrinsics_turntable_summary(camera_id))

    @app.route("/calibration/extrinsics/turntable/reset", methods=["POST"])
    def calibration_extrinsics_turntable_reset() -> Response:
        """Clear rotating-checkerboard extrinsics captures for one camera."""
        try:
            camera_id = _selected_camera_id()
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _extrinsics_turntable_lock:
            state = _extrinsics_turntable_state(camera_id)
            state["captures"] = []
            state["last_report"] = None
        return jsonify(_extrinsics_turntable_summary(camera_id))

    @app.route("/calibration/extrinsics/turntable/capture", methods=["POST"])
    def calibration_extrinsics_turntable_capture() -> Response:
        """Capture one checkerboard observation at the current turntable angle."""
        from scanner.calibration import checkerboard_capture_quality, draw_checkerboard_overlay
        from scanner.hardware import HardwareError

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        payload = request.get_json(silent=True) or {}
        source = payload if payload else request.values
        try:
            (
                camera_id,
                board_size,
                _square_mm,
                auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            ) = _parse_camera_board_payload()
            angle_deg = float(source.get("angle_deg", 0.0))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid turntable extrinsics payload: {exc}"}), 400

        try:
            frame, overlay, quality = _capture_checkerboard_candidate(
                camera_id,
                board_size,
                auto_bracket,
                calibration_exposure_us,
                calibration_gain,
            )
            if not quality.get("accepted"):
                return jsonify(
                    {
                        "error": quality.get("status", "checkerboard capture rejected"),
                        "quality": quality,
                        "preview_jpeg_base64": _encode_jpeg_base64(overlay),
                    }
                ), 422
            quality = checkerboard_capture_quality(frame, board_size)
            overlay = draw_checkerboard_overlay(frame, board_size, quality)
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 503
        except Exception as exc:
            logger.exception("Turntable extrinsics capture error")
            return jsonify({"error": f"Internal error during capture: {exc}"}), 500

        with _extrinsics_turntable_lock:
            state = _extrinsics_turntable_state(camera_id)
            state["captures"].append(
                {
                    "angle_deg": angle_deg,
                    "angle_rad": float(np.deg2rad(angle_deg)),
                    "image": frame,
                    "quality": {
                        key: value
                        for key, value in quality.items()
                        if key != "corners"
                    },
                    "preview_jpeg_base64": _encode_jpeg_base64(overlay, quality=70),
                }
            )
        return jsonify(_extrinsics_turntable_summary(camera_id))

    @app.route("/calibration/extrinsics/turntable/run", methods=["POST"])
    def calibration_extrinsics_turntable_run() -> Response:
        """Solve camera extrinsics from a checkerboard captured at multiple turntable angles."""
        from scanner.calibration import (
            CalibrationError,
            calibrate_camera_extrinsics_turntable,
            default_extrinsics_path,
            load_camera_calibration,
        )

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        payload = request.get_json(silent=True) or {}
        source = payload if payload else request.values
        try:
            (
                camera_id,
                board_size,
                square_mm,
                _auto_bracket,
                _calibration_exposure_us,
                _calibration_gain,
            ) = _parse_camera_board_payload()
            board_origin = _parse_vector3(source, "board_origin", (-100.0, 125.0, 0.0))
            board_col_axis = _parse_vector3(source, "board_col_axis", (1.0, 0.0, 0.0))
            board_row_axis = _parse_vector3(source, "board_row_axis", (0.0, -1.0, 0.0))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid turntable extrinsics payload: {exc}"}), 400

        with _extrinsics_turntable_lock:
            captures = list(_extrinsics_turntable_state(camera_id)["captures"])
        if len(captures) < 3:
            return jsonify({"error": f"At least 3 angle captures are required, got {len(captures)}"}), 422

        try:
            cam_cfg = _calibration_camera_config(camera_id)
            intrinsics_path = _project_path(cam_cfg.get("intrinsics_path"))
            if not intrinsics_path or not os.path.exists(intrinsics_path):
                return jsonify({"error": f"Camera {camera_id}: intrinsics file missing"}), 409
            camera_matrix, dist_coeffs = load_camera_calibration(intrinsics_path)
            output_path = _project_path(cam_cfg.get("extrinsics_path")) or default_extrinsics_path(camera_id)
            initial_rotation = None
            initial_translation = None
            try:
                from scanner.calibration.multi_camera import _look_at_extrinsics

                if isinstance(cam_cfg.get("extrinsics"), dict) and "position_mm" in cam_cfg["extrinsics"]:
                    initial_rotation, initial_translation = _look_at_extrinsics(cam_cfg["extrinsics"])
            except Exception:
                logger.warning(
                    "Camera %s extrinsics mechanical prior is invalid; solving without prior",
                    camera_id,
                    exc_info=True,
                )
            rotation_axis, rotation_axis_point = _platform_rotation_config()
            observations = [
                {
                    "image": item["image"],
                    "angle_rad": item["angle_rad"],
                }
                for item in captures
            ]
            rotation, translation, report = calibrate_camera_extrinsics_turntable(
                observations,
                camera_matrix,
                dist_coeffs,
                board_size,
                square_mm,
                board_origin,
                board_col_axis,
                board_row_axis,
                rotation_axis=rotation_axis,
                rotation_axis_point_mm=rotation_axis_point,
                output_path=output_path,
                initial_camera_to_platform_rotation=initial_rotation,
                initial_camera_to_platform_translation=initial_translation,
            )
            with _extrinsics_turntable_lock:
                _extrinsics_turntable_state(camera_id)["last_report"] = report
            return jsonify(
                {
                    "status": "ok",
                    "camera": camera_id,
                    "output_path": output_path,
                    "intrinsics_path": intrinsics_path,
                    "rotation_matrix": rotation.tolist(),
                    "translation_mm": translation.tolist(),
                    "report": report,
                }
            )
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.exception("Turntable extrinsics calibration error")
            return jsonify(
                {"error": f"Internal error during turntable extrinsics calibration: {exc}"}
            ), 500

    @app.route("/calibration/camera", methods=["POST"])
    def calibration_camera() -> Response:
        """Accept uploaded checkerboard images and run camera calibration."""
        import cv2  # type: ignore[import]
        import numpy as np
        from scanner.calibration import CalibrationError, calibrate_camera_with_report

        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No images uploaded"}), 400

        images: list[np.ndarray] = []
        for f in files:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

        if not images:
            return jsonify({"error": "Could not decode any uploaded images"}), 400

        try:
            (
                camera_id,
                board_size,
                square_mm,
                _auto_bracket,
                _calibration_exposure_us,
                _calibration_gain,
            ) = _parse_camera_board_payload()
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid checkerboard payload: {exc}"}), 400

        try:
            cam_cfg = _calibration_camera_config(camera_id)
            output_path = _project_path(cam_cfg.get("intrinsics_path"))
            camera_matrix, dist_coeffs, report = calibrate_camera_with_report(
                images,
                board_size=board_size,
                square_size_mm=square_mm,
                output_path=output_path,
            )
            return jsonify(
                {
                    "status": "ok",
                    "camera": camera_id,
                    "output_path": output_path,
                    "fx": float(camera_matrix[0, 0]),
                    "fy": float(camera_matrix[1, 1]),
                    "cx": float(camera_matrix[0, 2]),
                    "cy": float(camera_matrix[1, 2]),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "report": report,
                }
            )
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.exception("Camera calibration error")
            return jsonify({"error": f"Internal error during calibration: {exc}"}), 500

    @app.route("/calibration/laser", methods=["POST"])
    def calibration_laser() -> Response:
        """Run laser plane calibration using the current camera calibration."""
        import cv2  # type: ignore[import]
        import numpy as np
        from scanner.calibration import (
            CalibrationError,
            approximate_camera_intrinsics,
            calibrate_laser_plane,
            collect_laser_points_platform_z,
            fit_laser_plane_points,
            load_camera_calibration,
        )
        from scanner.calibration.multi_camera import _load_extrinsics

        try:
            camera_id = _selected_camera_id()
            cam_cfg = _calibration_camera_config(camera_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        files = request.files.getlist("images")
        distances_raw = request.form.get("distances_mm", "")
        plane_mode = str(request.form.get("plane_mode", "platform_z")).strip()

        if not files or not distances_raw:
            return jsonify({"error": "images and distances_mm are required"}), 400

        try:
            distances = [float(d.strip()) for d in distances_raw.split(",") if d.strip()]
        except ValueError as exc:
            return jsonify({"error": f"Invalid distances_mm: {exc}"}), 400

        images: list[np.ndarray] = []
        for f in files:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

        if len(images) != len(distances):
            return jsonify(
                {
                    "error": (
                        f"Got {len(images)} images but {len(distances)} distances — "
                        "must match"
                    )
                }
            ), 400

        calib_cfg = settings.get("calibration", {})
        focal_scale = float(cam_cfg.get("approx_focal_scale", calib_cfg.get("approx_focal_scale", 1.25)))
        resolution = cam_cfg.get("resolution", settings.get("camera", {}).get("resolution", [640, 480]))
        try:
            cam_res = (int(resolution[0]), int(resolution[1]))
        except Exception:
            cam_res = (640, 480)

        intrinsics_path = _project_path(cam_cfg.get("intrinsics_path"))
        try:
            if intrinsics_path and os.path.exists(intrinsics_path):
                camera_matrix, dist_coeffs = load_camera_calibration(intrinsics_path)
            else:
                camera_matrix, dist_coeffs = approximate_camera_intrinsics(
                    cam_res, focal_scale=focal_scale
                )
        except (CalibrationError, ValueError) as exc:
            return jsonify({"error": f"Camera intrinsics unavailable: {exc}"}), 409

        try:
            global_output_path = _project_path(
                settings.get("calibration", {}).get("global_laser_plane_path")
                or settings.get("laser", {}).get("plane_path")
                or "config/laser_plane.yaml"
            )
            default_threshold, mask_rects, sampling = _camera_processing_config(camera_id)
            proc_cfg = settings.get("processing", {})
            min_px = int(proc_cfg.get("min_line_pixels", 5))
            extraction_mode = str(proc_cfg.get("extraction_mode", "row_mean"))
            if plane_mode == "camera_depth":
                output_path = _project_path(cam_cfg.get("laser_plane_path"))
                plane = calibrate_laser_plane(
                    images,
                    distances,
                    camera_matrix,
                    dist_coeffs,
                    output_path=output_path,
                    threshold=default_threshold,
                    min_pixels=max(1, min_px),
                    mode=extraction_mode,
                    mask_rects=mask_rects,
                    camera_id=camera_id,
                    **sampling,
                )
            elif plane_mode == "platform_z":
                cam_rot, cam_trans = _load_extrinsics(cam_cfg)
                points = collect_laser_points_platform_z(
                    images,
                    distances,
                    camera_matrix,
                    dist_coeffs,
                    cam_rot,
                    cam_trans,
                    threshold=default_threshold,
                    min_pixels=max(1, min_px),
                    mode=extraction_mode,
                    mask_rects=mask_rects,
                    camera_id=camera_id,
                    **sampling,
                )
                output_path = global_output_path
                plane = fit_laser_plane_points(points, output_path=output_path)
            else:
                return jsonify({"error": f"Unknown laser calibration plane_mode: {plane_mode}"}), 400
            return jsonify(
                {
                    "status": "ok",
                    "camera": camera_id,
                    "plane_mode": plane_mode,
                    "output_path": output_path,
                    "intrinsics_path": intrinsics_path,
                    "used_approx_intrinsics": not bool(intrinsics_path and os.path.exists(intrinsics_path)),
                    "plane": plane.tolist(),
                }
            )
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.error("Laser calibration error: %s", exc)
            return jsonify({"error": "Internal error during laser calibration"}), 500

    @app.route("/calibration/laser/session")
    def calibration_laser_session() -> Response:
        """Return guided vertical-board laser calibration session."""
        return jsonify(_laser_calib_summary())

    @app.route("/calibration/laser/reset", methods=["POST"])
    def calibration_laser_reset() -> Response:
        """Clear guided laser calibration captures."""
        with _laser_calib_lock:
            _laser_calib_session["captures"] = []
            _laser_calib_session["last_result"] = None
        return jsonify(_laser_calib_summary())

    @app.route("/calibration/laser/capture", methods=["POST"])
    def calibration_laser_capture() -> Response:
        """Capture both cameras for one vertical-board platform Z position."""
        import cv2
        from scanner.hardware import HardwareError, camera_capture_all, laser_set
        from scanner.processing import extract_laser_line

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        payload = request.get_json(silent=True) or {}
        try:
            z_mm = float(payload.get("z_mm"))
            threshold_override = payload.get("threshold")
            min_px = int(payload.get("min_pixels", settings.get("processing", {}).get("min_line_pixels", 5)))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid laser capture payload: {exc}"}), 400
        min_px = max(1, min(4096, min_px))
        extraction_mode = str(settings.get("processing", {}).get("extraction_mode", "row_mean"))

        try:
            laser_set(True)
            time.sleep(float(settings.get("laser", {}).get("warmup_ms", 50)) / 1000.0)
            frames = camera_capture_all()
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 503
        finally:
            try:
                laser_set(False)
            except Exception:
                pass

        previews: dict = {}
        metrics: dict = {}
        stored_frames: dict = {}
        for camera_id, frame in frames.items():
            default_threshold, mask_rects, sampling = _camera_processing_config(camera_id)
            threshold = int(threshold_override if threshold_override is not None else default_threshold)
            threshold = max(0, min(255, threshold))
            line = extract_laser_line(
                frame,
                threshold=threshold,
                min_pixels=min_px,
                subpixel=True,
                mode=extraction_mode,
                camera_id=camera_id,
                mask_rects=mask_rects,
                **sampling,
            )
            overlay = frame.copy()
            for i in range(line.shape[0]):
                col, row = int(round(line[i, 0])), int(round(line[i, 1]))
                cv2.circle(overlay, (col, row), 2, (0, 0, 255), -1)
            green = frame[:, :, 1]
            previews[camera_id] = _encode_jpeg_base64(overlay, quality=70)
            metrics[camera_id] = {
                "detected_points": int(line.shape[0]),
                "green_max": int(green.max()),
                "green_mean": float(green.mean()),
                "saturated_pct": float((green >= 250).mean() * 100.0),
                "threshold": threshold,
                "min_pixels": min_px,
            }
            stored_frames[camera_id] = frame

        with _laser_calib_lock:
            _laser_calib_session["captures"].append(
                {
                    "z_mm": z_mm,
                    "frames": stored_frames,
                    "previews": previews,
                    "metrics": metrics,
                }
            )
        return jsonify(_laser_calib_summary())

    @app.route("/calibration/laser/run", methods=["POST"])
    def calibration_laser_run() -> Response:
        """Calibrate one shared platform-frame laser plane from guided captures."""
        import numpy as np

        from scanner.calibration import (
            CalibrationError,
            approximate_camera_intrinsics,
            camera_ids,
            collect_laser_points_platform_z,
            fit_laser_plane_points,
            load_camera_calibration,
        )
        from scanner.calibration.multi_camera import _load_extrinsics

        with _laser_calib_lock:
            captures = list(_laser_calib_session["captures"])
        if len(captures) < 3:
            return jsonify({"error": f"At least 3 Z captures are required, got {len(captures)}"}), 422

        proc_cfg = settings.get("processing", {})
        min_px = max(1, int(proc_cfg.get("min_line_pixels", 5)))
        extraction_mode = str(proc_cfg.get("extraction_mode", "row_mean"))
        results = {}
        all_points = []
        for camera_id in camera_ids(settings):
            cam_cfg = _calibration_camera_config(camera_id)
            images = [item["frames"][camera_id] for item in captures if camera_id in item["frames"]]
            z_values = [float(item["z_mm"]) for item in captures if camera_id in item["frames"]]
            if len(images) < 3:
                return jsonify({"error": f"Camera {camera_id}: at least 3 captures are required"}), 422

            calib_cfg = settings.get("calibration", {})
            focal_scale = float(cam_cfg.get("approx_focal_scale", calib_cfg.get("approx_focal_scale", 1.25)))
            resolution = cam_cfg.get("resolution", settings.get("camera", {}).get("resolution", [640, 480]))
            cam_res = (int(resolution[0]), int(resolution[1]))
            intrinsics_path = _project_path(cam_cfg.get("intrinsics_path"))
            try:
                if intrinsics_path and os.path.exists(intrinsics_path):
                    camera_matrix, dist_coeffs = load_camera_calibration(intrinsics_path)
                else:
                    camera_matrix, dist_coeffs = approximate_camera_intrinsics(
                        cam_res, focal_scale=focal_scale
                    )
                cam_rot, cam_trans = _load_extrinsics(cam_cfg)
                default_threshold, mask_rects, sampling = _camera_processing_config(camera_id)
                points = collect_laser_points_platform_z(
                    images,
                    z_values,
                    camera_matrix,
                    dist_coeffs,
                    cam_rot,
                    cam_trans,
                    threshold=default_threshold,
                    min_pixels=min_px,
                    mode=extraction_mode,
                    mask_rects=mask_rects,
                    camera_id=camera_id,
                    **sampling,
                )
            except CalibrationError as exc:
                return jsonify({"error": f"Camera {camera_id}: {exc}"}), 422
            except Exception as exc:
                logger.exception("Guided laser calibration error for %s", camera_id)
                return jsonify({"error": f"Camera {camera_id}: {exc}"}), 500

            results[camera_id] = {
                "intrinsics_path": intrinsics_path,
                "z_values": z_values,
                "points": int(points.shape[0]),
            }
            all_points.append(points)

        try:
            calib_cfg = settings.get("calibration", {})
            output_path = _project_path(
                calib_cfg.get("global_laser_plane_path")
                or settings.get("laser", {}).get("plane_path")
                or "config/laser_plane.yaml"
            )
            plane = fit_laser_plane_points(np.vstack(all_points), output_path=output_path)
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.exception("Global guided laser calibration error")
            return jsonify({"error": str(exc)}), 500

        result = {
            "status": "ok",
            "plane_mode": "platform_z_global",
            "output_path": output_path,
            "plane": plane.tolist(),
            "results": results,
        }
        with _laser_calib_lock:
            _laser_calib_session["last_result"] = result
        return jsonify(result)

    @app.route("/calibration/laser/test", methods=["POST"])
    def calibration_laser_test() -> Response:
        """Capture one laser test frame and return extraction overlay + metrics."""
        import cv2
        from scanner.hardware import HardwareError, camera_capture, laser_set
        from scanner.processing import extract_laser_line

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        proc_cfg = settings.get("processing", {})
        payload = request.get_json(silent=True) or {}
        camera_id = payload.get("camera")
        try:
            default_threshold, mask_rects, sampling = _camera_processing_config(camera_id)
            threshold = int(payload.get("threshold", default_threshold))
            min_px = int(payload.get("min_pixels", proc_cfg.get("min_line_pixels", 15)))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid laser test payload: {exc}"}), 400
        threshold = max(0, min(255, threshold))
        min_px = max(1, min(4096, min_px))
        extraction_mode = str(proc_cfg.get("extraction_mode", "row_mean"))

        try:
            laser_set(True)
            time.sleep(float(settings.get("laser", {}).get("warmup_ms", 50)) / 1000.0)
            frame = camera_capture(camera_id)
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 503
        finally:
            try:
                laser_set(False)
            except Exception:
                pass

        line = extract_laser_line(
            frame,
            threshold=threshold,
            min_pixels=min_px,
            subpixel=True,
            mode=extraction_mode,
            camera_id=camera_id,
            mask_rects=mask_rects,
            **sampling,
        )
        overlay = frame.copy()
        for i in range(line.shape[0]):
            col, row = int(round(line[i, 0])), int(round(line[i, 1]))
            cv2.circle(overlay, (col, row), 2, (0, 0, 255), -1)

        green = frame[:, :, 1]
        saturated_pct = float((green >= 250).mean() * 100.0)
        return jsonify(
            {
                "status": "ok",
                "detected_points": int(line.shape[0]),
                "green_max": int(green.max()),
                "green_mean": float(green.mean()),
                "saturated_pct": saturated_pct,
                "threshold": threshold,
                "min_pixels": min_px,
                "preview_jpeg_base64": _encode_jpeg_base64(overlay),
            }
        )

    # ------------------------------------------------------------------ #
    # Routes — USB export
    # ------------------------------------------------------------------ #

    @app.route("/usb/drives")
    def usb_drives() -> Response:
        """List detected USB drives."""
        from scanner.interface.usb import list_usb_drives

        drives = list_usb_drives()
        return jsonify(drives)

    @app.route("/usb/export", methods=["POST"])
    def usb_export() -> Response:
        """Copy the last scan file to a USB drive."""
        from scanner.interface.usb import copy_to_usb

        data = request.get_json(silent=True) or {}
        mountpoint = data.get("mountpoint", "")

        with _scan_lock:
            source = _scan_state.get("last_file")

        if not source or not os.path.exists(source):
            return jsonify({"error": "No scan file available"}), 404

        try:
            dest = copy_to_usb(source, mountpoint)
            return jsonify({"status": "ok", "path": dest})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            logger.error("USB copy error: %s", exc)
            return jsonify({"error": f"Copy failed: {exc}"}), 500

    return app


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg_path = str(
        Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
    )
    settings: dict = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh) or {}

    iface_cfg = settings.get("interface", {})
    host = iface_cfg.get("web_host", "0.0.0.0")
    port = int(iface_cfg.get("web_port", 5000))

    app = create_app(cfg_path)
    logger.info("Starting scanner web interface on http://%s:%d", host, port)
    app.run(host=host, port=port, threaded=True)
