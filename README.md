# PongBot – Real-Time Ping-Pong Paddle Tracker

A desktop app that uses your webcam to detect, track, and estimate the 3-D
pose of a standard ping-pong paddle in real time, then renders a virtual
copy of the paddle in a live OpenGL window.

---

## Quick Start

```bash
cd pongbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

If your webcam is not device 0, edit `CAMERA_INDEX` in `config.py`.

---

## Project Structure

```
pongbot/
├── main.py                        Entry point & pipeline glue
├── config.py                      All tunable parameters
├── requirements.txt
├── README.md
├── camera/
│   └── capture.py                 Background-thread camera reader
├── detection/
│   ├── detector.py                Colour seg + contour + ellipse fitting
│   └── pose_estimator.py          solvePnP 3-D pose estimation
├── tracking/
│   └── tracker.py                 Kalman + EMA smoothing, re-acquisition
├── visualization/
│   ├── camera_view.py             Qt camera-feed widget with CV overlays
│   └── paddle_3d_widget.py        QOpenGLWidget – 3-D paddle renderer
└── ui/
    └── main_window.py             Split-panel window + controls
```

---

## How Detection Works

1. **Colour segmentation (HSV)**
   - Red rubber: two hue bands (0–10° and 165–180°) with high saturation.
   - Black rubber: any hue, very low saturation **and** low value.
   - Each mask is cleaned with morphological open + close.

2. **Contour analysis**
   - The red and black masks are OR-ed to find the full paddle-head region.
   - External contours are filtered by:
     - Area: 2 000 – 200 000 px²
     - Circularity: ≥ 0.35  (4π·area / perimeter²)
     - Aspect ratio: ≤ 3.5

3. **Ellipse fitting**
   - `cv2.fitEllipse` is called on the best contour (≥ 5 points).
   - The fitted ellipse provides a robust centre, axes, and tilt angle.

4. **Visible-side classification**
   - Red and black pixel counts inside the contour mask determine which
     rubber face is toward the camera.

---

## How Pose Estimation Works

- **Method**: `cv2.solvePnP` (EPNP solver) with Levenberg–Marquardt refinement.
- **Object points**: 8 equally-spaced points on the known-size model ellipse
  (150 mm wide × 165 mm tall), lying in the Z = 0 plane.
- **Image points**: the same 8 points sampled from the *fitted* image ellipse.
- **Output**:
  - `rvec` – Rodrigues rotation vector (object frame → camera frame)
  - `tvec` – translation in metres (camera coordinates)
  - **Distance** = ‖tvec‖
- **Coordinate flip**: OpenCV uses Y-down / Z-forward; OpenGL uses Y-up / Z-back.
  The rotation matrix is transformed as `R_gl = diag(1,−1,−1) · R_cv · diag(1,−1,−1)`
  before being fed to `glMultMatrixf`.

### Camera Calibration

Default intrinsics (`fx = fy = 600 px`, `cx = 320`, `cy = 240`) are rough
estimates for a 640×480 built-in webcam.  For better distance accuracy:

```python
import cv2, numpy as np, glob

imgs = [cv2.imread(p) for p in glob.glob("calib/*.jpg")]
obj_pts, img_pts = [], []  # fill with chessboard corners
ret, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (640, 480), None, None)
print(K, dist)  # paste into config.py
```

Then update `Config.FOCAL_LENGTH_X/Y`, `PRINCIPAL_POINT_X/Y`, and
`Config.DIST_COEFFS` in `config.py`.

---

## Smoothing

Two layers run in series:

| Layer | What it smooths | Parameter |
|---|---|---|
| OpenCV Kalman filter | 2-D centre + bounding-box size | `KALMAN_PROCESS_NOISE`, `KALMAN_MEASUREMENT_NOISE` |
| Exponential Moving Average | 3-D rvec + tvec | `SMOOTHING_ALPHA` (UI slider or config.py) |

Lower `SMOOTHING_ALPHA` → smoother but more latency.
A value of 0.35 is a good starting point for 30 fps.

---

## Controls (runtime, no restart needed)

| Control | Effect |
|---|---|
| Debug Mode | Toggle all CV overlays |
| Show Contours | Draw the detected contour outline |
| Show Pose Axes | Draw X/Y/Z arrows from paddle centre |
| Show Depth | Display estimated distance |
| Smoothing slider | Adjust EMA alpha |
| Black Max Sat / Val | Tune black-rubber detection in your lighting |

When the paddle is **not tracked**, left-click-drag on the 3-D window to
orbit the model manually.

---

## Limitations

- **Monocular depth** is scale-ambiguous. Accuracy depends on how well the
  assumed paddle dimensions match your real paddle and on camera calibration.
- **Black detection** is sensitive to ambient lighting; rooms with low light
  or strong shadows may cause false positives.
- **Fast motion** can cause brief tracking loss due to motion blur; re-
  acquisition kicks in after `MAX_LOST_FRAMES` (default 20) frames.
- **Pose ambiguity**: `solvePnP` on planar objects has a 180° flip ambiguity.
  The face-colour classification partially resolves this but may glitch when
  the paddle is edge-on.
- No temporal pose-filtering across the flip ambiguity (future work).

---

## Future Expansion (planned for Wii-style game)

The codebase is structured so you can add:

| Module | Description |
|---|---|
| `detection/ball_detector.py` | Orange/white ball colour + circularity |
| `detection/table_detector.py` | Planar quad detection for the table surface |
| `physics/collision.py` | Ball–paddle / ball–table collision events |
| `game/state_machine.py` | Score, serve, game logic |
| `ai/opponent.py` | AI-controlled second paddle |
| `network/multiplayer.py` | UDP state sync for two-player |

All components share the `Config` singleton and the tracker state dict
format, making integration straightforward.
