"""
ui/main_window.py – Main application window.

Layout
------
 ┌──────────────────┬──────────────────┬────────────┐
 │  Camera Feed     │  3-D Paddle View │  Controls  │
 │  (CameraView)    │  (Paddle3DWidget)│  (panel)   │
 └──────────────────┴──────────────────┴────────────┘

The controls panel lets you tweak detection thresholds, smoothing, and
debug overlays without restarting the app.
"""
from __future__ import annotations  # noqa: F401 – enables X|Y unions on Py 3.9

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QSlider, QSpinBox, QVBoxLayout,
    QWidget,
)

from config import Config
from visualization.camera_view import CameraView
from visualization.paddle_3d_widget import Paddle3DWidget


class MainWindow(QMainWindow):
    """Top-level window combining the camera feed, 3-D view, and controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PongBot – Paddle Tracker")
        self.setMinimumSize(1300, 560)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Left: Camera feed ─────────────────────────────────────────────────
        self.camera_view = CameraView()
        self.camera_view.setMinimumSize(640, 480)

        # ── Centre: 3-D widget ────────────────────────────────────────────────
        self.paddle_3d = Paddle3DWidget()
        self.paddle_3d.setMinimumSize(500, 480)

        # ── Right: Controls ───────────────────────────────────────────────────
        controls = self._build_controls()
        controls.setFixedWidth(240)

        root.addWidget(self.camera_view, 3)
        root.addWidget(self.paddle_3d, 3)
        root.addWidget(controls, 1)

    # ── Controls builder ──────────────────────────────────────────────────────

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("PongBot Controls")
        title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(self._debug_group())
        layout.addWidget(self._smoothing_group())
        layout.addWidget(self._hsv_group())
        layout.addWidget(self._status_group())
        layout.addStretch()
        return panel

    def _debug_group(self) -> QGroupBox:
        g = QGroupBox("Debug Overlays")
        v = QVBoxLayout(g)

        def make_chk(label: str, attr: str, checked: bool) -> QCheckBox:
            chk = QCheckBox(label)
            chk.setChecked(checked)
            chk.toggled.connect(lambda val, a=attr: setattr(Config, a, val))
            return chk

        v.addWidget(make_chk("Debug Mode",      "DEBUG_MODE",    Config.DEBUG_MODE))
        v.addWidget(make_chk("Show Contours",   "SHOW_CONTOURS", Config.SHOW_CONTOURS))
        v.addWidget(make_chk("Show Pose Axes",  "SHOW_AXES",     Config.SHOW_AXES))
        v.addWidget(make_chk("Show Depth",      "SHOW_DEPTH",    Config.SHOW_DEPTH))
        v.addWidget(make_chk("Show Hand",       "SHOW_HAND",    Config.SHOW_HAND))
        v.addWidget(make_chk("YOLO Detection",  "YOLO_ENABLED", Config.YOLO_ENABLED))
        v.addWidget(make_chk("Hand Skeleton",   "HAND_DETECTION_ENABLED",
                             Config.HAND_DETECTION_ENABLED))
        return g

    def _smoothing_group(self) -> QGroupBox:
        g = QGroupBox("Smoothing (EMA α)")
        f = QFormLayout(g)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 100)
        slider.setValue(int(Config.SMOOTHING_ALPHA * 100))
        lbl = QLabel(f"{Config.SMOOTHING_ALPHA:.2f}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

        def on_change(v: int) -> None:
            Config.SMOOTHING_ALPHA = v / 100.0
            lbl.setText(f"{Config.SMOOTHING_ALPHA:.2f}")

        slider.valueChanged.connect(on_change)
        f.addRow("Alpha:", slider)
        f.addRow("", lbl)

        tip = QLabel("Low = smoother / more lag\nHigh = responsive / jittery")
        tip.setStyleSheet("font-size: 10px; color: #888;")
        f.addRow(tip)
        return g

    def _hsv_group(self) -> QGroupBox:
        """Quick sliders for black-mask saturation ceiling and value ceiling."""
        g = QGroupBox("Black Rubber Thresholds")
        f = QFormLayout(g)

        def sat_slider() -> tuple[QSlider, QLabel]:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 255)
            s.setValue(int(Config.BLACK_HSV_UPPER[1]))
            l = QLabel(str(s.value()))
            l.setAlignment(Qt.AlignmentFlag.AlignRight)

            def on_sat(v: int) -> None:
                Config.BLACK_HSV_UPPER[1] = v
                l.setText(str(v))

            s.valueChanged.connect(on_sat)
            return s, l

        def val_slider() -> tuple[QSlider, QLabel]:
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 255)
            s.setValue(int(Config.BLACK_HSV_UPPER[2]))
            l = QLabel(str(s.value()))
            l.setAlignment(Qt.AlignmentFlag.AlignRight)

            def on_val(v: int) -> None:
                Config.BLACK_HSV_UPPER[2] = v
                l.setText(str(v))

            s.valueChanged.connect(on_val)
            return s, l

        ss, sl = sat_slider()
        vs, vl = val_slider()
        f.addRow("Max Sat:", ss)
        f.addRow("", sl)
        f.addRow("Max Val:", vs)
        f.addRow("", vl)

        tip = QLabel("Raise if black paddle is not detected\nin bright lighting.")
        tip.setStyleSheet("font-size: 10px; color: #888;")
        f.addRow(tip)
        return g

    def _status_group(self) -> QGroupBox:
        g = QGroupBox("Status")
        v = QVBoxLayout(g)
        self._status_lbl = QLabel("Waiting…")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        v.addWidget(self._status_lbl)
        return g

    # ── Called from main loop ─────────────────────────────────────────────────

    def update_status(self, state: dict | None, fps: float) -> None:
        if state is None:
            self._status_lbl.setText("No paddle detected")
            return

        side     = state.get("visible_side", "—")
        tracking = "YES" if state.get("is_tracking") else "NO"
        tvec     = state.get("tvec")
        dist     = f"{np.linalg.norm(tvec):.2f} m" if tvec is not None else "—"
        lost     = state.get("frames_lost", 0)

        self._status_lbl.setText(
            f"Tracking:  {tracking}\n"
            f"Side:      {side}\n"
            f"Distance:  {dist}\n"
            f"FPS:       {fps:.1f}\n"
            f"Lost:      {lost}f"
        )
