"""
calibrate.py – One-time camera calibration for PongBot.

Usage
-----
    python calibrate.py

Instructions
------------
1. Print (or display on a second monitor) a standard 9×6 chessboard pattern.
   A free PDF is at:  https://calib.io/pages/camera-calibration-pattern-generator
   Make sure the squares are physically measurable — the default here assumes
   25 mm squares (0.025 m).  Adjust SQUARE_SIZE below if yours differ.

2. Hold the chessboard in front of your webcam and press SPACE to capture a
   frame.  Aim for ~20 frames from different angles and distances.

3. Press Q (or close the window) when done.  The script runs calibration and
   writes  models/camera_calibration.json.

4. Re-run  python main.py  — Config.load_calibration() picks up the file
   automatically on import.

Tips for a good calibration
----------------------------
- Cover the full field of view; don't just hold the board in the centre.
- Tilt the board (roll, pitch, yaw) across captures.
- Keep the board flat and fully visible in each capture.
- 15–25 frames is typically sufficient; 30+ rarely helps.
"""
import json
import os
import sys

import cv2
import numpy as np

from config import Config

# ── Configuration ─────────────────────────────────────────────────────────────

CHESSBOARD_COLS = 9   # inner corners (columns)
CHESSBOARD_ROWS = 6   # inner corners (rows)
SQUARE_SIZE     = 0.025   # metres per square side

TARGET_CAPTURES = 20
CALIBRATION_DIR = "models"
OUTPUT_PATH     = os.path.join(CALIBRATION_DIR, "camera_calibration.json")

# ── Build 3-D object points once ──────────────────────────────────────────────

_board = (CHESSBOARD_COLS, CHESSBOARD_ROWS)
_objp  = np.zeros((CHESSBOARD_COLS * CHESSBOARD_ROWS, 3), dtype=np.float32)
_objp[:, :2] = np.mgrid[0:CHESSBOARD_COLS, 0:CHESSBOARD_ROWS].T.reshape(-1, 2)
_objp *= SQUARE_SIZE

_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def main() -> None:
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("ERROR: Cannot open camera. Check your camera index in config.py.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    print(f"\nCamera Calibration – PongBot")
    print(f"Target: {TARGET_CAPTURES} captures  |  Press SPACE to capture  |  Press Q to finish early\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: dropped frame")
            continue

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, _board, None)
        if found:
            cv2.drawChessboardCorners(display, _board, corners, found)

        n = len(obj_points)
        status = f"Captured: {n}/{TARGET_CAPTURES}   {'[BOARD FOUND]' if found else '[no board]'}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if found else (0, 100, 255),
                    2, cv2.LINE_AA)
        cv2.putText(display, "SPACE=capture  Q=finish", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("PongBot Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' '):
            if not found:
                print("  No chessboard detected – try again")
                continue
            # Sub-pixel refinement
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _criteria)
            obj_points.append(_objp)
            img_points.append(corners2)
            print(f"  Captured frame {len(obj_points)}/{TARGET_CAPTURES}")
            if len(obj_points) >= TARGET_CAPTURES:
                print(f"\nReached {TARGET_CAPTURES} captures – running calibration…")
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 6:
        sys.exit(f"ERROR: Need at least 6 captures for calibration (got {len(obj_points)}).")

    print("Running cv2.calibrateCamera …", flush=True)
    h, w = frame.shape[:2]
    rms, cam_mat, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    fx = float(cam_mat[0, 0])
    fy = float(cam_mat[1, 1])
    cx = float(cam_mat[0, 2])
    cy = float(cam_mat[1, 2])
    dist = dist_coeffs.flatten().tolist()

    print(f"\n── Results ──────────────────────────────")
    print(f"  RMS reprojection error : {rms:.4f} px  (< 1.0 is good)")
    print(f"  fx={fx:.2f}  fy={fy:.2f}")
    print(f"  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  dist: {[f'{v:.5f}' for v in dist]}")

    if rms > 2.0:
        print("\nWARNING: RMS error is high (>2 px).  Consider recalibrating with")
        print("  more diverse board angles and ensuring the board is kept flat.")

    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "dist_coeffs": dist, "rms": rms}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to  {OUTPUT_PATH}")
    print("Run  python main.py  and pose estimation will use your real camera intrinsics.\n")


if __name__ == "__main__":
    main()
