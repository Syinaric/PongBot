"""
webapp.py – PongBot browser interface.
Run:  python webapp.py
Then open http://localhost:8080 in your browser.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera.capture import CameraCapture
from config import Config
from detection.detector import PaddleDetector
from detection.hand_detector import HandDetector
from detection.pose_estimator import PoseEstimator
from detection.yolo_detector import YOLODetector
from tracking.tracker import PaddleTracker

app = Flask(__name__)

# ── Pipeline components ────────────────────────────────────────────────────────
camera   = CameraCapture(Config.CAMERA_INDEX)
yolo     = YOLODetector()
hand_det = HandDetector()
detector = PaddleDetector()
pose_est = PoseEstimator()
tracker  = PaddleTracker()

# ── Three independent buffers ──────────────────────────────────────────────────
# 1. Latest raw camera frame (written by capture, read by render + detection)
_raw_frame: np.ndarray | None = None
_raw_lock = threading.Lock()

# 2. Latest detection state (written by detection thread, read by render thread)
_det_state: dict = {}
_det_hands: list = []
_det_yolo_boxes: list = []
_det_lock = threading.Lock()

# 3. Latest JPEG for streaming (written by render thread, read by Flask)
_jpeg: bytes | None = None
_jpeg_lock = threading.Lock()

# 4. Status for /api/status
_status: dict = {"is_tracking": False, "frames_lost": 0, "visible_side": "",
                 "distance": None, "fps": 0.0, "hand_detected": False, "yolo_detections": 0}
_status_lock = threading.Lock()


# ── Overlay drawing (pure OpenCV, mirrors camera_view.py) ─────────────────────

def _draw_overlays(canvas: np.ndarray, state: dict, hands: list,
                   yolo_boxes: list, fps: float) -> np.ndarray:
    if Config.SHOW_HAND and Config.HAND_DETECTION_ENABLED and hands:
        try:
            hand_det.draw(canvas, hands)
        except Exception:
            pass

    if Config.DEBUG_MODE:
        for box in yolo_boxes:
            if len(box) >= 5:
                x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 230, 0), 1)
                cv2.putText(canvas, f"YOLO {conf:.0%}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 0), 1, cv2.LINE_AA)

        det  = state.get("detection")
        rvec = state.get("rvec")
        tvec = state.get("tvec")
        wrist_px = hands[0].index_mcp if hands else None

        if det:
            if Config.SHOW_CONTOURS and det.get("contour") is not None:
                cv2.drawContours(canvas, [det["contour"]], -1, (0, 255, 255), 2)
            if det.get("ellipse"):
                cv2.ellipse(canvas, det["ellipse"], (255, 200, 0), 2)
            cx, cy = int(det["center"][0]), int(det["center"][1])
            cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
            side = det.get("visible_side", "")
            if side:
                clr = (30, 30, 200) if side == "red" else (30, 30, 30)
                cv2.putText(canvas, side.upper(), (cx - 25, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2, cv2.LINE_AA)

        if rvec is not None and tvec is not None:
            if Config.SHOW_AXES:
                try:
                    pose_est.draw_axes(canvas, rvec, tvec, wrist_px=wrist_px)
                except Exception:
                    pass
            if Config.SHOW_DEPTH:
                dist = float(np.linalg.norm(tvec))
                cv2.putText(canvas, f"Dist: {dist:.2f} m",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 0), 2, cv2.LINE_AA)

        if state.get("is_tracking"):
            label, clr = "TRACKING", (0, 210, 0)
        elif state.get("frames_lost", 0) > 0:
            label, clr = f"LOST ({state['frames_lost']}f)", (0, 130, 255)
        else:
            label, clr = "SEARCHING...", (0, 80, 200)
        cv2.putText(canvas, label,
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"FPS: {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2, cv2.LINE_AA)
    return canvas


# ── Thread 1: capture — grabs the latest camera frame as fast as possible ─────

def _capture_loop() -> None:
    global _raw_frame
    while True:
        frame = camera.get_frame()
        if frame is not None:
            with _raw_lock:
                _raw_frame = frame
        time.sleep(0.005)


# ── Thread 2: detection — runs the slow ML pipeline on the latest raw frame ───

def _detection_loop() -> None:
    global _det_state, _det_hands, _det_yolo_boxes
    while True:
        with _raw_lock:
            frame = _raw_frame
        if frame is None:
            time.sleep(0.01)
            continue

        frame = frame.copy()

        try:
            hands      = hand_det.detect(frame) if Config.HAND_DETECTION_ENABLED else []
            yolo_boxes = yolo.detect(frame) if (Config.YOLO_ENABLED and yolo.available) else []

            if hands:
                h = hands[0]
                pts_x  = [h.wrist[0], h.index_mcp[0]]
                pts_y  = [h.wrist[1], h.index_mcp[1]]
                margin = 200
                fh, fw = frame.shape[:2]
                hx1 = max(0, min(pts_x) - margin)
                hy1 = max(0, min(pts_y) - margin)
                hx2 = min(fw, max(pts_x) + margin)
                hy2 = min(fh, max(pts_y) + margin)
                yolo_boxes = list(yolo_boxes) + [(hx1, hy1, hx2, hy2, 0.9)]

            detection = detector.detect(frame, yolo_boxes=yolo_boxes)

            rvec, tvec = None, None
            if detection is not None:
                rvec, tvec, ok = pose_est.estimate_pose(detection)
                if not ok:
                    rvec, tvec = None, None

            wrist_px = hands[0].index_mcp if hands else None
            if rvec is not None and tvec is not None and wrist_px is not None:
                rvec = pose_est.refine_rotation_with_handle_dir(rvec, tvec, wrist_px)

            state = tracker.update(detection, rvec, tvec)
            if state is None:
                state = {}

            dist = float(np.linalg.norm(tvec)) if tvec is not None else None

            with _det_lock:
                _det_state       = state
                _det_hands       = hands
                _det_yolo_boxes  = yolo_boxes

            with _status_lock:
                _status.update({
                    "is_tracking":    bool(state.get("is_tracking")),
                    "frames_lost":    int(state.get("frames_lost", 0)),
                    "visible_side":   state.get("visible_side") or "",
                    "distance":       round(dist, 3) if dist is not None else None,
                    "fps":            round(camera.fps, 1),
                    "hand_detected":  len(hands) > 0,
                    "yolo_detections": len(yolo_boxes),
                })
        except Exception as exc:
            print(f"[detection] {exc}")


# ── Thread 3: render — composites raw frame + detection results at ~30 fps ────

def _render_loop() -> None:
    global _jpeg
    interval = 1 / 30
    while True:
        try:
            t0 = time.monotonic()

            with _raw_lock:
                frame = _raw_frame
            if frame is None:
                time.sleep(interval)
                continue

            with _det_lock:
                state      = dict(_det_state) if _det_state else {}
                hands      = list(_det_hands)
                yolo_boxes = list(_det_yolo_boxes)

            canvas = _draw_overlays(frame.copy(), state, hands, yolo_boxes, camera.fps)
            _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
            with _jpeg_lock:
                _jpeg = buf.tobytes()

            elapsed = time.monotonic() - t0
            time.sleep(max(0, interval - elapsed))
        except Exception as exc:
            print(f"[render] {exc}")
            time.sleep(0.1)


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/snapshot")
def snapshot():
    """Single JPEG frame — browser JS polls this to drive the live feed."""
    with _jpeg_lock:
        frame = _jpeg
    if frame is None:
        return Response(status=204)
    return Response(
        frame,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.route("/api/status")
def api_status():
    with _status_lock:
        return jsonify(dict(_status))


@app.route("/api/pose")
def api_pose():
    with _det_lock:
        state = dict(_det_state) if _det_state else {}
    rvec = state.get("rvec")
    tvec = state.get("tvec")
    return jsonify({
        "is_tracking":  bool(state.get("is_tracking")),
        "visible_side": state.get("visible_side") or "red",
        "rvec": rvec.flatten().tolist() if rvec is not None else None,
        "tvec": tvec.flatten().tolist() if tvec is not None else None,
    })


@app.route("/api/config", methods=["GET"])
def api_config_get():
    return jsonify({
        "smoothing_alpha":        round(Config.SMOOTHING_ALPHA, 2),
        "yolo_enabled":           Config.YOLO_ENABLED,
        "yolo_confidence":        round(Config.YOLO_CONFIDENCE, 2),
        "hand_detection_enabled": Config.HAND_DETECTION_ENABLED,
        "debug_mode":             Config.DEBUG_MODE,
        "show_contours":          Config.SHOW_CONTOURS,
        "show_axes":              Config.SHOW_AXES,
        "show_depth":             Config.SHOW_DEPTH,
        "show_hand":              Config.SHOW_HAND,
        "black_sat_max":          int(Config.BLACK_HSV_UPPER[1]),
        "black_val_max":          int(Config.BLACK_HSV_UPPER[2]),
    })


@app.route("/api/config", methods=["POST"])
def api_config_set():
    data = request.get_json(force=True)
    mapping = {
        "smoothing_alpha":        (float, "SMOOTHING_ALPHA"),
        "yolo_enabled":           (bool,  "YOLO_ENABLED"),
        "yolo_confidence":        (float, "YOLO_CONFIDENCE"),
        "hand_detection_enabled": (bool,  "HAND_DETECTION_ENABLED"),
        "debug_mode":             (bool,  "DEBUG_MODE"),
        "show_contours":          (bool,  "SHOW_CONTOURS"),
        "show_axes":              (bool,  "SHOW_AXES"),
        "show_depth":             (bool,  "SHOW_DEPTH"),
        "show_hand":              (bool,  "SHOW_HAND"),
    }
    for key, (cast, attr) in mapping.items():
        if key in data:
            setattr(Config, attr, cast(data[key]))
    if "black_sat_max" in data:
        Config.BLACK_HSV_UPPER[1] = int(data["black_sat_max"])
    if "black_val_max" in data:
        Config.BLACK_HSV_UPPER[2] = int(data["black_val_max"])
    return jsonify({"ok": True})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    camera.start()

    for target, name in [
        (_capture_loop,   "capture"),
        (_detection_loop, "detection"),
        (_render_loop,    "render"),
    ]:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()

    print("\n  PongBot Web Interface")
    print("  Open http://localhost:8080 in your browser\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
