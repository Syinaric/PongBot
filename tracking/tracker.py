"""
tracking/tracker.py – Temporal smoothing and re-acquisition logic.

Two layers of smoothing
-----------------------
1. OpenCV KalmanFilter on the 2-D centre + bounding-box size.
   State vector: [cx, cy, w, h, vcx, vcy, vw, vh]  (constant-velocity model)
   Measurement:  [cx, cy, w, h]

2. Exponential Moving Average (EMA) on the 3-D pose vectors (rvec / tvec).
   Lower SMOOTHING_ALPHA → smoother, more latency.

Re-acquisition
--------------
If no detection arrives for MAX_LOST_FRAMES consecutive frames the tracker
resets fully.  While frames_lost < MAX_LOST_FRAMES the Kalman prediction is
used so the 3-D view keeps a reasonable last-known pose.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Config


class PaddleTracker:
    """Fuses detections over time and returns a smoothed state dict."""

    def __init__(self) -> None:
        self._kf = self._build_kalman()
        self._rvec_ema: np.ndarray | None = None
        self._tvec_ema: np.ndarray | None = None
        self._last_detection: dict | None = None

        self.frames_lost: int = 0
        self.frames_tracked: int = 0
        self.is_tracking: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detection: dict | None,
        rvec: np.ndarray | None,
        tvec: np.ndarray | None,
    ) -> dict | None:
        """
        Call once per frame.  Returns a state dict or None if tracker has
        fully reset and no detection is available.
        """
        if detection is None:
            return self._handle_lost()

        # ── Detection available ───────────────────────────────────────────────
        self.frames_lost = 0
        self.frames_tracked += 1
        self.is_tracking = True
        self._last_detection = detection

        # Update Kalman with measured 2-D state
        cx, cy = detection["center"]
        x, y, w, h = detection["bbox"]
        meas = np.array([[float(cx)], [float(cy)], [float(w)], [float(h)]], dtype=np.float32)
        self._kf.correct(meas)
        pred = self._kf.predict()

        # EMA on 3-D pose
        if rvec is not None and tvec is not None:
            self._rvec_ema, self._tvec_ema = self._ema(rvec, tvec)

        return self._build_state(detection, pred)

    def reset(self) -> None:
        """Force a full tracker reset (e.g. when camera changes)."""
        self._kf = self._build_kalman()
        self._rvec_ema = None
        self._tvec_ema = None
        self._last_detection = None
        self.frames_lost = 0
        self.frames_tracked = 0
        self.is_tracking = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _handle_lost(self) -> dict | None:
        self.frames_lost += 1
        if self.frames_lost > Config.MAX_LOST_FRAMES:
            # Full reset – paddle has been gone too long
            if self.is_tracking:
                self.reset()
            return None

        # Still within grace window: return a predicted state
        pred = self._kf.predict()
        return self._build_state(None, pred)

    def _ema(
        self, rvec: np.ndarray, tvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        a = Config.SMOOTHING_ALPHA
        if self._rvec_ema is None:
            return rvec.copy(), tvec.copy()
        r = a * rvec + (1.0 - a) * self._rvec_ema
        t = a * tvec + (1.0 - a) * self._tvec_ema
        return r, t

    def _build_state(self, detection: dict | None, kalman_pred: np.ndarray) -> dict:
        px = float(kalman_pred[0])
        py = float(kalman_pred[1])
        side = (
            detection["visible_side"]
            if detection
            else (self._last_detection["visible_side"] if self._last_detection else "red")
        )
        return {
            "center": (int(px), int(py)),
            "rvec": self._rvec_ema,
            "tvec": self._tvec_ema,
            "visible_side": side,
            "is_tracking": self.is_tracking,
            "frames_lost": self.frames_lost,
            "detection": detection,
        }

    @staticmethod
    def _build_kalman() -> cv2.KalmanFilter:
        """Constant-velocity Kalman on [cx, cy, w, h]."""
        kf = cv2.KalmanFilter(8, 4)
        # Transition: state propagates with constant velocity
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1.0

        # Measurement picks only the first 4 state components
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)

        kf.processNoiseCov = np.eye(8, dtype=np.float32) * Config.KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * Config.KALMAN_MEASUREMENT_NOISE
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        return kf
