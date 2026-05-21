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
import io
import json
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
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
    }
    _scan_lock = threading.Lock()
    _sse_queue: queue.Queue = queue.Queue(maxsize=50)
    _camera_calib_lock = threading.Lock()
    _camera_calib_session: dict = {
        "images": [],
        "captures": [],
        "last_report": None,
    }

    def _load_background_filter_config() -> dict:
        try:
            from scanner.calibration import load_background_filter

            return load_background_filter()
        except Exception as exc:
            logger.warning("Background filter config unreadable: %s", exc)
            return {
                "enabled": False,
                "crop_left_of_col": None,
                "background_line_max_col": None,
                "margin_px": 0,
                "threshold": None,
                "min_pixels": None,
                "extraction_mode": None,
                "captured_at": None,
            }

    def _background_crop_left_col() -> float | None:
        data = _load_background_filter_config()
        if not data.get("enabled"):
            return None
        value = data.get("crop_left_of_col")
        return None if value is None else float(value)

    def _camera_calib_summary() -> dict:
        with _camera_calib_lock:
            captures = list(_camera_calib_session["captures"])
            report = _camera_calib_session.get("last_report")
        return {
            "count": len(captures),
            "recommended_min": 12,
            "recommended_max": 20,
            "captures": captures,
            "last_report": report,
        }

    def _parse_camera_board_payload() -> tuple[tuple[int, int], float, bool]:
        payload = request.get_json(silent=True) or {}
        source = payload if payload else request.values
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
        return board_size, square_mm, auto_bracket

    def _encode_jpeg_base64(frame, quality: int = 85) -> str:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Could not encode JPEG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _capture_checkerboard_candidate(
        board_size: tuple[int, int],
        auto_bracket: bool,
    ) -> tuple:
        from scanner.calibration import checkerboard_capture_quality, draw_checkerboard_overlay
        from scanner.hardware import HardwareError, camera_capture, camera_set_exposure, laser_set

        cam_cfg = settings.get("camera", {})
        scan_exposure = int(cam_cfg.get("exposure_us", 1000))
        base_exposure = int(cam_cfg.get("calibration_exposure_us", max(5000, scan_exposure * 6)))
        gain = float(cam_cfg.get("gain", 1.0))
        exposures = [base_exposure]
        exposure_note = None
        if auto_bracket:
            exposures = [max(100, int(base_exposure * factor)) for factor in (0.5, 1.0, 2.0, 4.0)]

        with _camera_calib_lock:
            previous_poses = [c["pose"] for c in _camera_calib_session["captures"] if c.get("pose")]

        best = None
        try:
            laser_set(False)
            for exposure in exposures:
                try:
                    camera_set_exposure(exposure, gain)
                    time.sleep(0.08)
                except HardwareError as exc:
                    exposure_note = f"reglage exposition indisponible: {exc}"
                    if auto_bracket:
                        logger.warning("Calibration exposure bracketing unavailable: %s", exc)
                    exposures = [base_exposure]
                frame = camera_capture()
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
                camera_set_exposure(scan_exposure, gain)
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
        return render_template("index.html", scan_state=state)

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

            try:
                path = run_scan(settings, progress_callback=_cb, state_machine=_sm)
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
        return jsonify(state)

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
        import numpy as np
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
        import numpy as np
        from scanner.hardware.mock import MockCamera
        from scanner.processing import crop_laser_line, extract_laser_line

        angle_rad = float(request.args.get("angle", 0.0))
        cam_cfg = settings.get("camera", {"resolution": [640, 480]})
        cam = MockCamera(cam_cfg)
        cam.set_rotation_angle(angle_rad)
        frame = cam.capture()

        proc_cfg = settings.get("processing", {})
        threshold = int(proc_cfg.get("laser_threshold", 180))
        min_px = int(proc_cfg.get("min_line_pixels", 10))
        subpixel = bool(proc_cfg.get("subpixel", True))
        extraction_mode = str(proc_cfg.get("extraction_mode", "component_axis"))

        line = extract_laser_line(
            frame,
            threshold=threshold,
            min_pixels=min_px,
            subpixel=subpixel,
            mode=extraction_mode,
        )
        line = crop_laser_line(
            line,
            crop_left_of_col=_background_crop_left_col(),
            min_points=min_px,
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
        return render_template("manual.html", scan_state=state)

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
        from scanner.processing import crop_laser_line, extract_laser_line

        if not _manual_allowed():
            return Response(status=409)

        proc_cfg = settings.get("processing", {})
        default_threshold = int(proc_cfg.get("laser_threshold", 60))
        default_min_px = int(proc_cfg.get("min_line_pixels", 15))
        extraction_mode = str(proc_cfg.get("extraction_mode", "component_axis"))

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

        try:
            frame = camera_capture(request.args.get("camera"))
        except HardwareError:
            return Response(status=503)

        line = extract_laser_line(
            frame,
            threshold=threshold,
            min_pixels=min_px,
            subpixel=True,
            mode=extraction_mode,
        )
        line = crop_laser_line(
            line,
            crop_left_of_col=_background_crop_left_col(),
            min_points=min_px,
        )

        signal = frame[:, :, 1]  # green channel only
        gr_max = int(signal.max())
        gr_mean = float(signal.mean())

        overlay = frame.copy()
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
        resp.headers["Access-Control-Expose-Headers"] = (
            "X-Detected-Columns,X-GR-Max,X-GR-Mean,X-Threshold,X-Min-Pixels"
        )
        return resp

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
        return render_template(
            "calibration.html",
            use_checkerboard=use_checkerboard,
            background_filter=_load_background_filter_config(),
            camera_calib=_camera_calib_summary(),
        )

    @app.route("/calibration/camera/session")
    def calibration_camera_session() -> Response:
        """Return current in-memory checkerboard capture session."""
        return jsonify(_camera_calib_summary())

    @app.route("/calibration/camera/reset", methods=["POST"])
    def calibration_camera_reset() -> Response:
        """Clear captured checkerboard images."""
        with _camera_calib_lock:
            _camera_calib_session["images"] = []
            _camera_calib_session["captures"] = []
            _camera_calib_session["last_report"] = None
        return jsonify(_camera_calib_summary())

    @app.route("/calibration/camera/frame")
    def calibration_camera_frame() -> Response:
        """Return one live checkerboard frame with detection overlay."""
        import cv2
        from scanner.calibration import checkerboard_capture_quality, draw_checkerboard_overlay
        from scanner.hardware import HardwareError, camera_capture, laser_set

        if not _manual_allowed():
            return Response(status=409)
        try:
            board_size, _square_mm, _auto_bracket = _parse_camera_board_payload()
        except (TypeError, ValueError):
            board_size = (9, 6)

        try:
            laser_set(False)
            frame = camera_capture()
        except HardwareError:
            return Response(status=503)
        finally:
            try:
                laser_set(False)
            except Exception:
                pass

        with _camera_calib_lock:
            previous_poses = [c["pose"] for c in _camera_calib_session["captures"] if c.get("pose")]
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
            board_size, _square_mm, auto_bracket = _parse_camera_board_payload()
            frame, overlay, quality = _capture_checkerboard_candidate(board_size, auto_bracket)
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
                idx = len(_camera_calib_session["images"]) + 1
                _camera_calib_session["images"].append(frame)
                _camera_calib_session["captures"].append(
                    {
                        "index": idx,
                        "pose": quality.get("pose"),
                        "metrics": metrics,
                        "exposure_us": quality.get("exposure_us"),
                        "preview_jpeg_base64": preview,
                    }
                )

        result = _camera_calib_summary()
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

        if not bool(settings.get("calibration", {}).get("use_checkerboard", True)):
            return jsonify(
                {
                    "error": (
                        "Checkerboard calibration is disabled in settings "
                        "(calibration.use_checkerboard=false)."
                    )
                }
            ), 409
        try:
            board_size, square_mm, _auto_bracket = _parse_camera_board_payload()
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid checkerboard payload: {exc}"}), 400

        with _camera_calib_lock:
            images = list(_camera_calib_session["images"])
        if len(images) < 4:
            return jsonify({"error": f"At least 4 accepted images are required, got {len(images)}"}), 422

        try:
            camera_matrix, dist_coeffs, report = calibrate_camera_with_report(
                images, board_size=board_size, square_size_mm=square_mm
            )
            result = {
                "status": "ok",
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "dist_coeffs": dist_coeffs.tolist(),
                "report": report,
            }
            with _camera_calib_lock:
                _camera_calib_session["last_report"] = report
            return jsonify(result)
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.error("Guided camera calibration error: %s", exc)
            return jsonify({"error": "Internal error during calibration"}), 500

    @app.route("/calibration/camera", methods=["POST"])
    def calibration_camera() -> Response:
        """Accept uploaded checkerboard images and run camera calibration."""
        import cv2  # type: ignore[import]
        import numpy as np
        from scanner.calibration import CalibrationError, calibrate_camera_with_report

        if not bool(settings.get("calibration", {}).get("use_checkerboard", True)):
            return jsonify(
                {
                    "error": (
                        "Checkerboard calibration is disabled in settings "
                        "(calibration.use_checkerboard=false)."
                    )
                }
            ), 409

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

        board_size = (
            int(request.form.get("board_cols", 9)),
            int(request.form.get("board_rows", 6)),
        )
        square_mm = float(request.form.get("square_size_mm", 25.0))

        try:
            camera_matrix, dist_coeffs, report = calibrate_camera_with_report(
                images, board_size=board_size, square_size_mm=square_mm
            )
            return jsonify(
                {
                    "status": "ok",
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
            logger.error("Camera calibration error: %s", exc)
            return jsonify({"error": "Internal error during calibration"}), 500

    @app.route("/calibration/laser", methods=["POST"])
    def calibration_laser() -> Response:
        """Run laser plane calibration using the current camera calibration."""
        import cv2  # type: ignore[import]
        import numpy as np
        from scanner.calibration import (
            CalibrationError,
            approximate_camera_intrinsics,
            calibrate_laser_plane,
            load_camera_calibration,
        )

        files = request.files.getlist("images")
        distances_raw = request.form.get("distances_mm", "")

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
        use_checkerboard = bool(calib_cfg.get("use_checkerboard", True))
        focal_scale = float(calib_cfg.get("approx_focal_scale", 1.25))
        cam_cfg = settings.get("camera", {})
        resolution = cam_cfg.get("resolution", [640, 480])
        try:
            cam_res = (int(resolution[0]), int(resolution[1]))
        except Exception:
            cam_res = (640, 480)

        try:
            if use_checkerboard:
                camera_matrix, dist_coeffs = load_camera_calibration()
            else:
                camera_matrix, dist_coeffs = approximate_camera_intrinsics(
                    cam_res, focal_scale=focal_scale
                )
        except (CalibrationError, ValueError) as exc:
            return jsonify({"error": f"Camera intrinsics unavailable: {exc}"}), 409

        try:
            plane = calibrate_laser_plane(
                images,
                distances,
                camera_matrix,
                dist_coeffs,
                crop_left_of_col=_background_crop_left_col(),
            )
            return jsonify({"status": "ok", "plane": plane.tolist()})
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.error("Laser calibration error: %s", exc)
            return jsonify({"error": "Internal error during laser calibration"}), 500

    @app.route("/calibration/laser/test", methods=["POST"])
    def calibration_laser_test() -> Response:
        """Capture one laser test frame and return extraction overlay + metrics."""
        import cv2
        from scanner.hardware import HardwareError, camera_capture, laser_set
        from scanner.processing import crop_laser_line, extract_laser_line

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        proc_cfg = settings.get("processing", {})
        payload = request.get_json(silent=True) or {}
        try:
            threshold = int(payload.get("threshold", proc_cfg.get("laser_threshold", 60)))
            min_px = int(payload.get("min_pixels", proc_cfg.get("min_line_pixels", 15)))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid laser test payload: {exc}"}), 400
        threshold = max(0, min(255, threshold))
        min_px = max(1, min(4096, min_px))
        extraction_mode = str(proc_cfg.get("extraction_mode", "component_axis"))

        try:
            laser_set(True)
            time.sleep(float(settings.get("laser", {}).get("warmup_ms", 50)) / 1000.0)
            frame = camera_capture()
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
        )
        line = crop_laser_line(
            line,
            crop_left_of_col=_background_crop_left_col(),
            min_points=min_px,
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

    @app.route("/calibration/background-filter", methods=["POST"])
    def calibration_background_filter() -> Response:
        """Capture an empty frame and calibrate the left-image crop."""
        import cv2
        import numpy as np
        from scanner.calibration import save_background_filter
        from scanner.hardware import HardwareError, camera_capture, laser_set
        from scanner.processing import extract_laser_line

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        proc_cfg = settings.get("processing", {})
        default_threshold = int(proc_cfg.get("laser_threshold", 60))
        default_min_px = int(proc_cfg.get("min_line_pixels", 15))
        extraction_mode = str(proc_cfg.get("extraction_mode", "component_axis"))

        payload = request.get_json(silent=True) or {}
        try:
            threshold = int(payload.get("threshold", default_threshold))
            min_px = int(payload.get("min_pixels", default_min_px))
            margin_px = int(payload.get("margin_px", 6))
        except (TypeError, ValueError) as exc:
            return jsonify({"error": f"Invalid calibration payload: {exc}"}), 400

        threshold = max(0, min(255, threshold))
        min_px = max(1, min(4096, min_px))
        margin_px = max(0, min(200, margin_px))

        frame = None
        try:
            laser_set(True)
            frame = camera_capture()
        except HardwareError as exc:
            return jsonify({"error": str(exc)}), 500
        finally:
            try:
                laser_set(False)
            except Exception:
                pass

        if frame is None:
            return jsonify({"error": "Camera capture failed"}), 500

        line = extract_laser_line(
            frame,
            threshold=threshold,
            min_pixels=min_px,
            subpixel=True,
            mode=extraction_mode,
        )
        if line.shape[0] < min_px:
            return jsonify(
                {
                    "error": (
                        f"Ligne de fond introuvable: {line.shape[0]} points detectes "
                        f"(minimum requis: {min_px})"
                    )
                }
            ), 422

        background_col = float(np.max(line[:, 0]))
        crop_left_of_col = float(math.ceil(background_col + margin_px))
        saved = save_background_filter(
            crop_left_of_col=crop_left_of_col,
            background_line_max_col=background_col,
            margin_px=margin_px,
            threshold=threshold,
            min_pixels=min_px,
            extraction_mode=extraction_mode,
        )

        overlay = frame.copy()
        for i in range(line.shape[0]):
            col, row = int(round(line[i, 0])), int(round(line[i, 1]))
            cv2.circle(overlay, (col, row), 2, (0, 0, 255), -1)
        cutoff_x = int(round(crop_left_of_col))
        cv2.line(
            overlay,
            (cutoff_x, 0),
            (cutoff_x, overlay.shape[0] - 1),
            (255, 255, 0),
            2,
        )
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return jsonify({"error": "Could not encode calibration preview"}), 500

        saved["preview_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return jsonify(saved)

    @app.route("/calibration/background-filter/disable", methods=["POST"])
    def calibration_background_filter_disable() -> Response:
        """Disable the calibrated left-image crop."""
        from scanner.calibration import disable_background_filter

        if not _manual_allowed():
            return jsonify({"error": "Calibration disabled while scan is running"}), 409

        return jsonify(disable_background_filter())

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
