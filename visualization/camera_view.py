"""
visualization/camera_view.py – Live camera feed with debug overlays.
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from config import Config


class CameraView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._label.setText("Waiting for camera…")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def update_frame(
        self,
        frame: np.ndarray,
        state: dict | None,
        pose_estimator,
        fps: float = 0.0,
        yolo_boxes: list | None = None,
        hand_detector=None,
        hands: list | None = None,
        face_boxes: list | None = None,
    ) -> None:
        if frame is None:
            return

        canvas = frame.copy()

        # Use index_mcp (blade-handle junction) for the green axis arrow — matches
        # the same anchor used in main.py for spin-ambiguity correction.
        wrist_px = hands[0].index_mcp if hands else None

        if hand_detector and hands and Config.SHOW_HAND:
            hand_detector.draw(canvas, hands)

        if Config.DEBUG_MODE:
            if yolo_boxes:
                for x1, y1, x2, y2, conf in yolo_boxes:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 230, 0), 1)
                    cv2.putText(canvas, f"YOLO {conf:.0%}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 0), 1, cv2.LINE_AA)

            if state is not None:
                self._draw_overlays(canvas, state, pose_estimator, wrist_px)

        cv2.putText(canvas, f"FPS: {fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2, cv2.LINE_AA)

        self._show(canvas)

    def _draw_overlays(self, canvas: np.ndarray, state: dict, pe, wrist_px=None) -> None:
        det  = state.get("detection")
        rvec = state.get("rvec")
        tvec = state.get("tvec")

        if det:
            if Config.SHOW_CONTOURS and det.get("contour") is not None:
                cv2.drawContours(canvas, [det["contour"]], -1, (0, 255, 255), 2)
            if det.get("ellipse"):
                cv2.ellipse(canvas, det["ellipse"], (255, 200, 0), 2)

            cx, cy = det["center"]
            cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)

            side = det.get("visible_side", "")
            if side:
                clr = (30, 30, 200) if side == "red" else (30, 30, 30)
                cv2.putText(canvas, side.upper(), (cx - 25, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2, cv2.LINE_AA)

        if rvec is not None and tvec is not None:
            if Config.SHOW_AXES:
                try:
                    pe.draw_axes(canvas, rvec, tvec, wrist_px=wrist_px)
                except Exception:
                    pass
            if Config.SHOW_DEPTH:
                dist = float(np.linalg.norm(tvec))
                cv2.putText(canvas, f"Dist: {dist:.2f} m",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 230, 0), 2, cv2.LINE_AA)

        if state.get("is_tracking"):
            label, clr = "TRACKING", (0, 210, 0)
        elif state.get("frames_lost", 0) > 0:
            label, clr = f"LOST ({state['frames_lost']}f)", (0, 130, 255)
        else:
            label, clr = "SEARCHING…", (0, 80, 200)
        cv2.putText(canvas, label,
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2, cv2.LINE_AA)

    def _show(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._label.width(), self._label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pix)
