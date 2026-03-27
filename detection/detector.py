"""
detection/detector.py – Colour + YOLO paddle detector.

1. If YOLO returns boxes, run colour segmentation inside those ROIs.
2. Otherwise run colour segmentation on the whole frame.
3. Filter contours by area, circularity and solidity.
4. Pick the best contour (highest score) and return it.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Config


class DetectionResult(dict):
    pass


class PaddleDetector:
    _ROI_MARGIN = 30

    def __init__(self) -> None:
        self._k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def detect(
        self,
        frame: np.ndarray,
        yolo_boxes: list | None = None,
    ) -> DetectionResult | None:
        h, w = frame.shape[:2]
        red_mask, black_mask = self._colour_masks(frame)
        combined = cv2.bitwise_or(red_mask, black_mask)

        candidates: list[np.ndarray] = []

        if yolo_boxes:
            for x1, y1, x2, y2, _ in yolo_boxes:
                rx1 = max(0, x1 - self._ROI_MARGIN)
                ry1 = max(0, y1 - self._ROI_MARGIN)
                rx2 = min(w, x2 + self._ROI_MARGIN)
                ry2 = min(h, y2 + self._ROI_MARGIN)
                roi = combined[ry1:ry2, rx1:rx2]
                cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    candidates.append(c + np.array([[[rx1, ry1]]]))

        if not candidates:
            cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = list(cnts)

        best: DetectionResult | None = None
        best_score = -1.0

        for cnt in candidates:
            result = self._analyse(cnt, frame, red_mask, black_mask)
            if result is not None and result["score"] > best_score:
                best_score = result["score"]
                best = result

        return best

    def _colour_masks(self, frame: np.ndarray):
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        r1  = cv2.inRange(hsv, Config.RED_HSV_LOWER1, Config.RED_HSV_UPPER1)
        r2  = cv2.inRange(hsv, Config.RED_HSV_LOWER2, Config.RED_HSV_UPPER2)
        red = cv2.bitwise_or(r1, r2)
        blk = cv2.inRange(hsv, Config.BLACK_HSV_LOWER, Config.BLACK_HSV_UPPER)

        red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  self._k5)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, self._k11)
        blk = cv2.morphologyEx(blk, cv2.MORPH_OPEN,  self._k5)
        blk = cv2.morphologyEx(blk, cv2.MORPH_CLOSE, self._k11)
        return red, blk

    def _analyse(self, cnt, frame, red_mask, black_mask) -> DetectionResult | None:
        area = cv2.contourArea(cnt)
        if not (Config.MIN_PADDLE_AREA <= area <= Config.MAX_PADDLE_AREA):
            return None

        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            return None
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < Config.MIN_CIRCULARITY:
            return None

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if area / max(hull_area, 1) < Config.MIN_SOLIDITY:
            return None

        x, y, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) / max(min(bw, bh), 1) > Config.MAX_ASPECT_RATIO:
            return None
        cx, cy = x + bw // 2, y + bh // 2

        ellipse = None
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                pass

        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(roi_mask, [cnt], -1, 255, cv2.FILLED)
        red_px = int(cv2.countNonZero(cv2.bitwise_and(red_mask,   roi_mask)))
        blk_px = int(cv2.countNonZero(cv2.bitwise_and(black_mask, roi_mask)))
        total  = red_px + blk_px
        if total < 200:
            return None

        side      = "red" if red_px >= blk_px else "black"
        side_conf = max(red_px, blk_px) / total
        score     = circ * min(area / 8_000.0, 1.0) * side_conf

        return DetectionResult(
            contour=cnt,
            center=(cx, cy),
            bbox=(x, y, bw, bh),
            ellipse=ellipse,
            area=area,
            visible_side=side,
            red_ratio=red_px / total,
            black_ratio=blk_px / total,
            side_confidence=side_conf,
            score=score,
        )
