"""
detection/yolo_detector.py – YOLOv8 paddle detector wrapper.

Priority order
--------------
1. Custom model at Config.YOLO_MODEL_PATH  (e.g. models/paddle.pt)
   – expects class 0 = "paddle"
2. COCO YOLOv8n fallback: class 38 = tennis racket (rough proxy)

Usage
-----
    from detection.yolo_detector import YOLODetector
    yd = YOLODetector()
    boxes = yd.detect(frame)   # list of (x1, y1, x2, y2, conf)

To get a custom ping-pong paddle model
---------------------------------------
Train or download one from Roboflow (search "table tennis paddle" or
"ping pong paddle" on universe.roboflow.com), export as YOLOv8 PyTorch,
and place the .pt file at Config.YOLO_MODEL_PATH.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from config import Config

# (x1, y1, x2, y2, confidence)
BBox = Tuple[int, int, int, int, float]


class YOLODetector:
    """Thin wrapper around an ultralytics YOLO model."""

    def __init__(self) -> None:
        self._model = None
        self._class_ids: list[int] = []   # class indices to keep
        self._custom: bool = False
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Run inference on *frame* (BGR).
        Returns a list of (x1, y1, x2, y2, conf) boxes, sorted by confidence.
        """
        if self._model is None:
            return []
        try:
            results = self._model(
                frame,
                imgsz=Config.YOLO_IMGSZ,
                conf=Config.YOLO_CONFIDENCE,
                verbose=False,
            )
        except Exception:
            return []

        boxes: List[BBox] = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if self._class_ids and cls not in self._class_ids:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, conf))

        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes

    def best_box(self, frame: np.ndarray) -> BBox | None:
        """Return highest-confidence detection or None."""
        boxes = self.detect(frame)
        return boxes[0] if boxes else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            from ultralytics import YOLO  # local import to avoid startup cost
        except ImportError:
            print("[YOLO] ultralytics not installed – YOLO disabled.")
            return

        custom_path = Config.YOLO_MODEL_PATH
        if os.path.isfile(custom_path):
            print(f"[YOLO] Loading custom model: {custom_path}")
            try:
                self._model = YOLO(custom_path)
                self._class_ids = [0]   # assume single class "paddle"
                self._custom = True
                print("[YOLO] Custom model loaded OK.")
                return
            except Exception as e:
                print(f"[YOLO] Custom model failed ({e}), falling back to COCO.")

        # Fall back to YOLOv8n pretrained on COCO
        print("[YOLO] No custom model found. Using YOLOv8n COCO (tennis-racket proxy).")
        print("[YOLO]   → For better results, add a paddle-specific model to models/paddle.pt")
        try:
            self._model = YOLO("yolov8n.pt")   # auto-downloads ~6 MB on first run
            self._class_ids = [Config.YOLO_COCO_FALLBACK_CLASS]
            self._custom = False
        except Exception as e:
            print(f"[YOLO] Could not load fallback model: {e}")
            self._model = None
