"""
config.py – Global configuration for PongBot paddle tracker.

All tunable parameters live here so the rest of the codebase stays
import-free of magic numbers. The UI also mutates these class attributes
at runtime (no restart required).

Camera calibration
------------------
Run  python calibrate.py  once to measure your camera's real intrinsics.
The script writes  models/camera_calibration.json  which is loaded below,
overriding the generic defaults.
"""
import json
import os

import numpy as np


class Config:
    # ── Camera ───────────────────────────────────────────────────────────────
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 30

    # ── Paddle physical dimensions (metres, ITTF approximate) ────────────────
    PADDLE_HEAD_WIDTH: float = 0.150
    PADDLE_HEAD_HEIGHT: float = 0.165
    PADDLE_HANDLE_LENGTH: float = 0.085
    PADDLE_HANDLE_WIDTH: float = 0.025
    PADDLE_THICKNESS: float = 0.006

    # ── HSV colour thresholds ─────────────────────────────────────────────────
    # Red rubber – moderate saturation floor; we can afford to be looser
    # now that we only check inside the hand-predicted ROI.
    RED_HSV_LOWER1: np.ndarray = np.array([0,   100,  50], dtype=np.uint8)
    RED_HSV_UPPER1: np.ndarray = np.array([12,  255, 255], dtype=np.uint8)
    RED_HSV_LOWER2: np.ndarray = np.array([163, 100,  50], dtype=np.uint8)
    RED_HSV_UPPER2: np.ndarray = np.array([180, 255, 255], dtype=np.uint8)

    # Black rubber – reasonably tight so very dark clothing in the ROI
    # doesn't overwhelm; user can widen via UI sliders.
    BLACK_HSV_LOWER: np.ndarray = np.array([0,   0,   0], dtype=np.uint8)
    BLACK_HSV_UPPER: np.ndarray = np.array([180, 80,  60], dtype=np.uint8)

    # ── Detection shape constraints ───────────────────────────────────────────
    MIN_PADDLE_AREA: int = 2_500
    MAX_PADDLE_AREA: int = 180_000
    MIN_CIRCULARITY: float = 0.50   # paddle head is nearly circular; hair is not
    MIN_SOLIDITY: float = 0.78      # convexity check — paddle head is convex
    MAX_ASPECT_RATIO: float = 3.0   # reject very elongated blobs (arms, neck)

    # Minimum fraction of pixels inside the contour that must be paddle colour.
    # Raises the bar over blobs that are mostly background noise.
    MIN_COLOUR_FILL: float = 0.25

    # ── Spatial rejection filters ─────────────────────────────────────────────
    # When a hand is detected, reject any contour whose centre is more than
    # this many pixels from the wrist.  Eliminates face / hair hits entirely.
    MAX_WRIST_DISTANCE: int = 380

    # Face exclusion: only applied in full-frame fallback (no hand detected).
    # Disabled by default because the hand-guided ROI already avoids the face.
    FACE_EXCLUSION_ENABLED: bool = False
    FACE_EXCLUSION_MARGIN: int = 55

    # ── Camera intrinsics (pixels) ────────────────────────────────────────────
    FOCAL_LENGTH_X: float = 600.0
    FOCAL_LENGTH_Y: float = 600.0
    PRINCIPAL_POINT_X: float = 320.0
    PRINCIPAL_POINT_Y: float = 240.0
    DIST_COEFFS: np.ndarray = np.zeros(5, dtype=np.float64)

    # ── Smoothing ─────────────────────────────────────────────────────────────
    SMOOTHING_ALPHA: float = 0.50
    KALMAN_PROCESS_NOISE: float = 5e-2
    KALMAN_MEASUREMENT_NOISE: float = 5e-2
    MAX_LOST_FRAMES: int = 20

    # ── YOLO detector ─────────────────────────────────────────────────────────
    YOLO_MODEL_PATH: str = "models/paddle.pt"
    YOLO_ENABLED: bool = True
    YOLO_CONFIDENCE: float = 0.40
    YOLO_IMGSZ: int = 416
    YOLO_COCO_FALLBACK_CLASS: int = 38   # tennis racket

    # ── Hand detector ─────────────────────────────────────────────────────────
    HAND_DETECTION_ENABLED: bool = True
    HAND_MAX_HANDS: int = 1
    HAND_DETECTION_CONFIDENCE: float = 0.60
    HAND_TRACKING_CONFIDENCE: float = 0.50

    # ── Debug overlays ────────────────────────────────────────────────────────
    DEBUG_MODE: bool = True
    SHOW_CONTOURS: bool = True
    SHOW_AXES: bool = True
    SHOW_DEPTH: bool = True
    SHOW_HAND: bool = True
    SHOW_FACE_EXCLUSION: bool = True   # draw face exclusion boxes in debug mode

    # ── Load camera calibration from file (if it exists) ─────────────────────
    @classmethod
    def load_calibration(cls, path: str = "models/camera_calibration.json") -> bool:
        """Load intrinsics from calibrate.py output.  Returns True on success."""
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as f:
                d = json.load(f)
            cls.FOCAL_LENGTH_X    = float(d["fx"])
            cls.FOCAL_LENGTH_Y    = float(d["fy"])
            cls.PRINCIPAL_POINT_X = float(d["cx"])
            cls.PRINCIPAL_POINT_Y = float(d["cy"])
            cls.DIST_COEFFS       = np.array(d["dist_coeffs"], dtype=np.float64)
            return True
        except Exception:
            return False


# Load calibration at import time so all modules see the real intrinsics.
Config.load_calibration()
