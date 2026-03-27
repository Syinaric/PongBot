"""
camera/capture.py – Thread-safe webcam capture with frame buffering.

The capture runs in a daemon thread so the main Qt event loop is never
blocked waiting for a frame.  Call start() once, then get_frame() each
tick to retrieve the latest image without blocking.
"""
from __future__ import annotations

import threading
import time
import cv2

from config import Config


class CameraCapture:
    """Continuous camera reader that buffers the most-recent frame."""

    def __init__(self, camera_index: int | None = None):
        self.camera_index = camera_index if camera_index is not None else Config.CAMERA_INDEX
        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # FPS counter
        self.fps: float = 0.0
        self._fps_count: int = 0
        self._fps_t0: float = time.time()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the capture device and begin the reader thread."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Try a different CAMERA_INDEX in config.py."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, Config.FPS_TARGET)
        # Reduce internal buffer to minimise latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="CameraThread", daemon=True)
        self._thread.start()

    def get_frame(self):
        """Return a copy of the latest frame (or None if not yet available)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Signal the reader thread to stop and release the device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        self._cap = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
            self._tick_fps()

    def _tick_fps(self) -> None:
        self._fps_count += 1
        now = time.time()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self.fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_t0 = now
