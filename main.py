"""
main.py – PongBot entry point.
"""
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from camera.capture import CameraCapture
from config import Config
from detection.detector import PaddleDetector
from detection.hand_detector import HandDetector
from detection.pose_estimator import PoseEstimator
from detection.yolo_detector import YOLODetector
from tracking.tracker import PaddleTracker
from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    camera   = CameraCapture(Config.CAMERA_INDEX)
    yolo     = YOLODetector()
    hand_det = HandDetector()
    detector = PaddleDetector()
    pose_est = PoseEstimator()
    tracker  = PaddleTracker()

    window = MainWindow()
    window.show()

    try:
        camera.start()
    except RuntimeError as exc:
        QMessageBox.critical(window, "Camera Error", str(exc))
        sys.exit(1)

    def process() -> None:
        frame = camera.get_frame()
        if frame is None:
            return

        # Hand skeleton overlay (visual only — not used for detection)
        hands = hand_det.detect(frame) if Config.HAND_DETECTION_ENABLED else []

        # YOLO hint boxes
        yolo_boxes = yolo.detect(frame) if (Config.YOLO_ENABLED and yolo.available) else []

        # Add a hand-guided ROI so the detector always searches near the hand,
        # even if YOLO misses the paddle.
        if hands:
            h = hands[0]
            pts_x = [h.wrist[0], h.index_mcp[0]]
            pts_y = [h.wrist[1], h.index_mcp[1]]
            margin = 200
            fh, fw = frame.shape[:2]
            hx1 = max(0, min(pts_x) - margin)
            hy1 = max(0, min(pts_y) - margin)
            hx2 = min(fw, max(pts_x) + margin)
            hy2 = min(fh, max(pts_y) + margin)
            yolo_boxes = list(yolo_boxes) + [(hx1, hy1, hx2, hy2, 0.9)]

        # Paddle detection
        detection = detector.detect(frame, yolo_boxes=yolo_boxes)

        # Pose estimation
        rvec, tvec = None, None
        if detection is not None:
            rvec, tvec, ok = pose_est.estimate_pose(detection)
            if not ok:
                rvec, tvec = None, None

        # Resolve spin ambiguity using index_mcp (blade-handle junction).
        # index_mcp (landmark 5) sits at the base of the index finger, right
        # where the handle meets the blade — a geometrically tighter anchor
        # than the wrist for computing the handle direction vector.
        wrist_px = hands[0].index_mcp if hands else None
        if rvec is not None and tvec is not None and wrist_px is not None:
            rvec = pose_est.refine_rotation_with_handle_dir(rvec, tvec, wrist_px)

        # Tracker
        state = tracker.update(detection, rvec, tvec)

        # UI
        window.camera_view.update_frame(
            frame, state, pose_est,
            fps=camera.fps,
            yolo_boxes=yolo_boxes,
            hand_detector=hand_det,
            hands=hands,
        )
        window.paddle_3d.update_pose(state)
        window.update_status(state, camera.fps)

    timer = QTimer()
    timer.timeout.connect(process)
    timer.start(33)

    app.aboutToQuit.connect(lambda: (camera.stop(), hand_det.close()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
