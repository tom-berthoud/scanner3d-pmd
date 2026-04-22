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

import io
import json
import logging
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
        path = "/tmp/scan_frames/latest.jpg"
        if not os.path.exists(path):
            return Response(status=404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/scan/frame/<int:step>")
    def scan_frame_step(step: int) -> Response:
        """Return the JPEG frame captured at the given step index."""
        path = f"/tmp/scan_frames/frame_{step:03d}.jpg"
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
        from scanner.processing import extract_laser_line

        angle_rad = float(request.args.get("angle", 0.0))
        cam_cfg = settings.get("camera", {"resolution": [640, 480]})
        cam = MockCamera(cam_cfg)
        cam.set_rotation_angle(angle_rad)
        frame = cam.capture()

        proc_cfg = settings.get("processing", {})
        threshold = int(proc_cfg.get("laser_threshold", 180))
        min_px = int(proc_cfg.get("min_line_pixels", 10))
        subpixel = bool(proc_cfg.get("subpixel", True))

        line = extract_laser_line(frame, threshold=threshold, min_pixels=min_px, subpixel=subpixel)

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
            frame = camera_capture()
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
        default_threshold = int(proc_cfg.get("laser_threshold", 60))
        default_min_px = int(proc_cfg.get("min_line_pixels", 15))

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
            frame = camera_capture()
        except HardwareError:
            return Response(status=503)

        line = extract_laser_line(
            frame, threshold=threshold, min_pixels=min_px, subpixel=True
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
        )

    @app.route("/calibration/camera", methods=["POST"])
    def calibration_camera() -> Response:
        """Accept uploaded checkerboard images and run camera calibration."""
        import cv2  # type: ignore[import]
        import numpy as np
        from scanner.calibration import calibrate_camera, CalibrationError

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
            camera_matrix, dist_coeffs = calibrate_camera(
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
            plane = calibrate_laser_plane(images, distances, camera_matrix, dist_coeffs)
            return jsonify({"status": "ok", "plane": plane.tolist()})
        except CalibrationError as exc:
            return jsonify({"error": str(exc)}), 422
        except Exception as exc:
            logger.error("Laser calibration error: %s", exc)
            return jsonify({"error": "Internal error during laser calibration"}), 500

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
