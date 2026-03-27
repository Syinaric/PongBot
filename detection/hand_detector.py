"""
detection/hand_detector.py – MediaPipe Hands wrapper (Tasks API, v0.10+).

The HandLandmarker model is downloaded automatically on first run to
models/hand_landmarker.task.

Key landmarks used
------------------
  0  WRIST           → handle base (anchor for paddle detection)
  5  INDEX_FINGER_MCP → handle/blade junction
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import Config

Point2D = Tuple[int, int]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe Tasks API (import paths for 0.10+)
from mediapipe.tasks import python as _mp_python          # noqa: E402
from mediapipe.tasks.python import vision as _mp_vision   # noqa: E402

# Hand skeleton connections (21-node graph)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]


@dataclass
class HandResult:
    wrist:        Point2D
    index_mcp:    Point2D
    landmarks_px: List[Point2D]
    handedness:   str          # "Left" or "Right"


def _ensure_model() -> str:
    """Download the hand-landmarker task file if not already present."""
    path = os.path.normpath(_MODEL_PATH)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("[Hand] Downloading hand_landmarker.task (≈28 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("[Hand] Model saved to", path)
    return path


class HandDetector:
    """
    Wraps MediaPipe HandLandmarker (Tasks API).
    Runs in VIDEO mode for low-latency per-frame inference.
    """

    _WRIST     = 0
    _INDEX_MCP = 5

    def __init__(self) -> None:
        model_path = _ensure_model()
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=_mp_vision.RunningMode.VIDEO,
            num_hands=Config.HAND_MAX_HANDS,
            min_hand_detection_confidence=Config.HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=Config.HAND_TRACKING_CONFIDENCE,
        )
        self._lm = _mp_vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()   # epoch for timestamp_ms

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[HandResult]:
        """
        Run detection on *frame* (BGR).  Returns a list of HandResult.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)  # type: ignore[attr-defined]
        ts_ms  = int((time.monotonic() - self._t0) * 1000)

        try:
            result = self._lm.detect_for_video(mp_img, ts_ms)
        except Exception:
            return []

        if not result.hand_landmarks:
            return []

        h, w = frame.shape[:2]
        out: List[HandResult] = []

        for i, lm_list in enumerate(result.hand_landmarks):
            px = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            side = "Right"
            if result.handedness and i < len(result.handedness):
                side = result.handedness[i][0].display_name
            out.append(HandResult(
                wrist=px[self._WRIST],
                index_mcp=px[self._INDEX_MCP],
                landmarks_px=px,
                handedness=side,
            ))
        return out

    def draw(self, frame: np.ndarray, hands: List[HandResult]) -> None:
        """Draw hand skeleton on *frame* in-place."""
        if not Config.SHOW_HAND or not hands:
            return
        for hand in hands:
            # Skeleton edges
            for a, b in _CONNECTIONS:
                cv2.line(frame, hand.landmarks_px[a], hand.landmarks_px[b],
                         (160, 160, 160), 1, cv2.LINE_AA)
            # All joints
            for pt in hand.landmarks_px:
                cv2.circle(frame, pt, 3, (200, 200, 200), -1, cv2.LINE_AA)
            # Highlight wrist (green) and index-MCP (yellow)
            cv2.circle(frame, hand.wrist,     9, (0, 255, 80),  -1, cv2.LINE_AA)
            cv2.circle(frame, hand.index_mcp, 7, (0, 220, 255), -1, cv2.LINE_AA)
            wx, wy = hand.wrist
            cv2.putText(frame, f"hand ({hand.handedness})",
                        (wx + 10, wy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 255, 80), 1, cv2.LINE_AA)

    def wrist_point(self, hands: List[HandResult]) -> Point2D | None:
        return hands[0].wrist if hands else None

    def close(self) -> None:
        self._lm.close()
