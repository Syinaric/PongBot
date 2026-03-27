"""
detection/hand_paddle_predictor.py – Predict paddle head position from hand.

The hand is the ground truth.  The paddle is a rigid extension of the hand:
the handle is gripped, so the paddle head is a known physical distance beyond
the wrist in the direction the hand is pointing.

Geometry (2-D image space)
--------------------------
  • Direction vector  d  = normalise(avg_MCP − WRIST)
      The four knuckles (MCP 5, 9, 13, 17) define the "heel" of the palm.
      The vector from wrist to avg-MCP points along the hand / handle.

  • Scale s (metres/pixel) is estimated from the palm segment:
      |avg_MCP − WRIST| in pixels  ↔  PALM_LENGTH_M (≈ 95 mm for adults).
      This gives a rough scale without needing camera calibration.

  • Predicted paddle head centre (pixels):
      P = WRIST + d × (handle_length + head_height/2) / s

  • Predicted radius:
      r = head_width/2 / s

All values are clamped to the frame bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config

# Approximate adult palm length (wrist to MCP row) in metres
_PALM_M = 0.095

# MCP landmark indices
_MCP_INDICES = [5, 9, 13, 17]


@dataclass
class PaddlePrediction:
    """Geometric prediction of where the paddle head should be."""
    center: tuple[int, int]   # predicted head centre (pixels)
    radius: int               # predicted head radius  (pixels)
    direction: np.ndarray     # unit vector wrist → head (image plane)
    scale: float              # metres per pixel (estimated from hand)
    wrist: tuple[int, int]    # wrist anchor used for the prediction


class HandPaddlePredictor:
    """
    Converts MediaPipe hand landmarks into a PaddlePrediction.
    Pure geometry – no image data needed here.
    """

    def predict(
        self,
        hand,           # detection.hand_detector.HandResult
        frame_h: int,
        frame_w: int,
    ) -> PaddlePrediction | None:
        """
        Return a PaddlePrediction or None if the hand data is unusable.
        """
        wrist_px = np.array(hand.wrist, dtype=float)
        mcp_px   = np.array(
            [hand.landmarks_px[i] for i in _MCP_INDICES], dtype=float
        )
        avg_mcp = mcp_px.mean(axis=0)

        # Hand direction vector
        palm_vec = avg_mcp - wrist_px
        palm_len = float(np.linalg.norm(palm_vec))
        if palm_len < 10:   # hand too small / too close to edge
            return None

        direction = palm_vec / palm_len

        # Scale estimate
        scale = _PALM_M / palm_len   # m/px

        # Physical distance from wrist to paddle head centre
        dist_m = Config.PADDLE_HANDLE_LENGTH + Config.PADDLE_HEAD_HEIGHT / 2.0
        dist_px = dist_m / scale

        # Predicted head centre
        cx = int(wrist_px[0] + direction[0] * dist_px)
        cy = int(wrist_px[1] + direction[1] * dist_px)
        cx = max(0, min(frame_w - 1, cx))
        cy = max(0, min(frame_h - 1, cy))

        # Predicted head radius
        radius = max(10, int((Config.PADDLE_HEAD_WIDTH / 2.0) / scale))
        radius = min(radius, min(frame_w, frame_h) // 3)   # sanity cap

        return PaddlePrediction(
            center=(cx, cy),
            radius=radius,
            direction=direction,
            scale=scale,
            wrist=(int(wrist_px[0]), int(wrist_px[1])),
        )
