"""
detection/face_excluder.py – Face exclusion zones using OpenCV Haar cascades.

The Haar cascade XML files ship with every opencv-python install so no extra
download is needed.  Detected face bounding boxes (expanded by a configurable
margin) are passed to the paddle detector which hard-rejects any contour whose
centre falls inside one.

This acts as a safety net when the hand detector cannot see the wrist (e.g.
wrist out of frame, hand occluded).  When the wrist IS visible the hand-
proximity constraint is already stronger, so face exclusion is a secondary
fallback rather than the primary gate.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Config

# Both frontal and profile cascades for robustness
_FRONTAL_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_PROFILE_CASCADE = cv2.data.haarcascades + "haarcascade_profileface.xml"

# Exclusion box type: (x1, y1, x2, y2)
FaceBox = tuple[int, int, int, int]


class FaceExcluder:
    """
    Detects faces in a frame and returns exclusion boxes.
    Runs every frame; designed to be fast (Haar is CPU-only, ~2–5 ms).
    """

    def __init__(self) -> None:
        self._frontal = cv2.CascadeClassifier(_FRONTAL_CASCADE)
        self._profile = cv2.CascadeClassifier(_PROFILE_CASCADE)

        if self._frontal.empty():
            print("[FaceExcluder] WARNING: frontal cascade not found.")
        if self._profile.empty():
            print("[FaceExcluder] WARNING: profile cascade not found.")

        # Downscale factor for detection speed (detect on smaller image)
        self._scale = 0.5
        # Cache last result to avoid every-frame detection (refresh every N frames)
        self._cache: list[FaceBox] = []
        self._cache_ttl: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_boxes(self, frame: np.ndarray) -> list[FaceBox]:
        """
        Return a list of (x1, y1, x2, y2) face exclusion boxes
        (already expanded by FACE_EXCLUSION_MARGIN).

        Results are cached for 5 frames to keep overhead minimal.
        """
        if not Config.FACE_EXCLUSION_ENABLED:
            return []

        self._cache_ttl -= 1
        if self._cache_ttl > 0:
            return self._cache

        self._cache = self._detect(frame)
        self._cache_ttl = 5   # refresh every 5 frames (~6 Hz at 30 fps)
        return self._cache

    def contour_in_face(self, cx: int, cy: int, boxes: list[FaceBox]) -> bool:
        """Return True if point (cx, cy) falls inside any face exclusion box."""
        for x1, y1, x2, y2 in boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> list[FaceBox]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * self._scale), int(h * self._scale)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        boxes: list[FaceBox] = []
        m = int(Config.FACE_EXCLUSION_MARGIN / self._scale)  # margin in small-scale px

        for cascade in (self._frontal, self._profile):
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(50, 50),
            )
            if len(faces) == 0:
                continue
            for (fx, fy, fw, fh) in faces:
                # Scale back to full resolution and add margin
                x1 = max(0, int((fx - m) / self._scale))
                y1 = max(0, int((fy - m) / self._scale))
                x2 = min(w, int((fx + fw + m) / self._scale))
                y2 = min(h, int((fy + fh + m) / self._scale))
                boxes.append((x1, y1, x2, y2))

        return self._merge_overlapping(boxes, w, h)

    @staticmethod
    def _merge_overlapping(boxes: list[FaceBox], w: int, h: int) -> list[FaceBox]:
        """Union any overlapping boxes to avoid double-exclusion gaps."""
        if not boxes:
            return []
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        merged: list[FaceBox] = []
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            merged.append((x, y, x + bw, y + bh))
        return merged
