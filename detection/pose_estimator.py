"""
detection/pose_estimator.py – Monocular 3-D pose estimation for the paddle.

Method
------
We model the paddle head as a planar ellipse with known real-world semi-axes
(hw × hh metres).  Eight equally-spaced 3-D object points are placed on that
ellipse at z = 0.  The matching 2-D image points come from the fitted ellipse
returned by the detector.

cv2.solvePnP (EPNP) solves the Perspective-n-Point problem and returns:
  rvec  – Rodrigues rotation vector  (paddle frame → camera frame)
  tvec  – translation vector in metres (camera coords)

Distance is simply ||tvec||.

Coordinate conventions
-----------------------
  OpenCV camera frame:  X right, Y down, Z into scene.
  The 3-D axes drawn on the camera view use the same convention.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import Config


class PoseEstimator:
    """Estimates rvec / tvec from a detection result using solvePnP."""

    # Number of ellipse sample points used for PnP
    N_PTS = 8

    def __init__(self) -> None:
        self._update_intrinsics()
        self._build_model_points()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_intrinsics(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist: np.ndarray | None = None,
    ) -> None:
        """Hot-swap camera intrinsics at runtime."""
        Config.FOCAL_LENGTH_X = fx
        Config.FOCAL_LENGTH_Y = fy
        Config.PRINCIPAL_POINT_X = cx
        Config.PRINCIPAL_POINT_Y = cy
        if dist is not None:
            Config.DIST_COEFFS = dist
        self._update_intrinsics()

    def estimate_pose(
        self, detection: dict
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        """
        Return (rvec, tvec, success) from a detection dict.

        Tries the fitted ellipse first; falls back to the axis-aligned
        bounding box if the ellipse is unavailable.
        """
        if detection.get("ellipse") is not None:
            return self._from_ellipse(detection["ellipse"])
        return self._from_bbox(detection["bbox"])

    def get_distance(self, tvec: np.ndarray | None) -> float | None:
        if tvec is None:
            return None
        return float(np.linalg.norm(tvec))

    def refine_rotation_with_handle_dir(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        wrist_px: tuple[int, int],
    ) -> np.ndarray:
        """
        Fix the spin ambiguity in rvec using the known wrist position.

        solvePnP recovers the paddle's tilt/tilt-axis well from the ellipse
        shape but cannot determine the spin around the paddle's own normal
        (Z-axis) because the head is rotationally symmetric.  The wrist gives
        us the handle direction, which resolves that ambiguity.

        Coordinate geometry
        -------------------
        In the paddle model  +Y points toward the top of the head (away from
        the handle), so the handle direction is  -Y  in model space.

        We unproject the wrist pixel to a 3D ray at the paddle's depth,
        giving an approximate wrist position in camera space.  The vector
        from the paddle centre (tvec) to that wrist position is the handle
        direction in camera space, i.e. the model's  -Y  axis expressed in
        camera coordinates.

        We then rebuild the rotation matrix keeping the Z column (paddle
        normal) from solvePnP and replacing the Y column with the corrected
        handle direction.  X = Y × Z closes the orthonormal basis.
        """
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        # Unproject wrist pixel → 3D ray at paddle depth
        depth = float(t[2])
        if depth < 0.01:
            return rvec
        wx = (wrist_px[0] - Config.PRINCIPAL_POINT_X) / Config.FOCAL_LENGTH_X
        wy = (wrist_px[1] - Config.PRINCIPAL_POINT_Y) / Config.FOCAL_LENGTH_Y
        wrist_3d = np.array([wx * depth, wy * depth, depth])

        # Handle direction in camera space: paddle centre → wrist
        handle_cam = wrist_3d - t
        handle_len = np.linalg.norm(handle_cam)
        if handle_len < 1e-6:
            return rvec
        handle_cam /= handle_len

        # The 3-D widget uses F @ R @ F (double-flip convention).
        # Under that convention R's Y column maps the *OpenCV-flipped*
        # model +Y (= original model −Y = handle direction) into camera space.
        # So y_cam must equal the handle direction, not its inverse.
        y_cam = handle_cam

        # Keep Z (paddle normal) from solvePnP; orthogonalise Y against it
        z_cam = R[:, 2].copy()
        y_cam = y_cam - np.dot(y_cam, z_cam) * z_cam
        y_norm = np.linalg.norm(y_cam)
        if y_norm < 1e-6:
            return rvec
        y_cam /= y_norm

        # Right-handed X = Y × Z
        x_cam = np.cross(y_cam, z_cam)
        x_norm = np.linalg.norm(x_cam)
        if x_norm < 1e-6:
            return rvec
        x_cam /= x_norm

        R_new = np.column_stack([x_cam, y_cam, z_cam])

        # Guarantee det = +1 (proper rotation)
        if np.linalg.det(R_new) < 0:
            x_cam = -x_cam
            R_new = np.column_stack([x_cam, y_cam, z_cam])

        rvec_new, _ = cv2.Rodrigues(R_new)
        return rvec_new

    def draw_axes(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        length: float = 0.05,
        wrist_px: tuple[int, int] | None = None,
    ) -> None:
        """Draw X (red), Y (green), Z (blue) axes on the frame.

        If *wrist_px* is provided the green Y arrow is replaced with a line
        from the projected paddle centre to the wrist, so it always points
        toward the user's hand regardless of paddle orientation.
        """
        axis_pts = np.float32(
            [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]]
        )
        try:
            projected, _ = cv2.projectPoints(
                axis_pts, rvec, tvec, self._cam_mat, self._dist
            )
        except cv2.error:
            return

        pts = projected.reshape(-1, 2).astype(int)
        o = tuple(pts[0])
        cv2.arrowedLine(frame, o, tuple(pts[1]), (0, 0, 255), 2, tipLength=0.2)   # X red

        # Green: toward wrist when available, otherwise default Y axis
        green_tip = wrist_px if wrist_px is not None else tuple(pts[2])
        cv2.arrowedLine(frame, o, green_tip, (0, 255, 0), 2, tipLength=0.2)       # Y green

        cv2.arrowedLine(frame, o, tuple(pts[3]), (255, 100, 0), 2, tipLength=0.2) # Z blue

    def project_model(
        self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> None:
        """Project the 3-D paddle outline onto the frame for debug."""
        try:
            pts2d, _ = cv2.projectPoints(
                self._obj_pts_rect, rvec, tvec, self._cam_mat, self._dist
            )
        except cv2.error:
            return
        pts2d = pts2d.reshape(-1, 2).astype(int)
        cv2.polylines(frame, [pts2d], True, (255, 255, 0), 1)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_intrinsics(self) -> None:
        self._cam_mat = np.array(
            [
                [Config.FOCAL_LENGTH_X, 0, Config.PRINCIPAL_POINT_X],
                [0, Config.FOCAL_LENGTH_Y, Config.PRINCIPAL_POINT_Y],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._dist = Config.DIST_COEFFS.copy()

    def _build_model_points(self) -> None:
        hw = Config.PADDLE_HEAD_WIDTH / 2.0
        hh = Config.PADDLE_HEAD_HEIGHT / 2.0

        # 8 equally-spaced points on the ellipse perimeter
        angles = np.linspace(0, 2 * np.pi, self.N_PTS, endpoint=False)
        self._obj_pts = np.array(
            [[hw * np.cos(a), hh * np.sin(a), 0.0] for a in angles],
            dtype=np.float32,
        )

        # 4-corner rectangle (fallback for bounding-box estimates)
        self._obj_pts_rect = np.array(
            [[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]],
            dtype=np.float32,
        )

    def _from_ellipse(self, ellipse) -> tuple:
        (cx, cy), (ma, mb), angle_deg = ellipse
        ra, rb = ma / 2.0, mb / 2.0
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        angles = np.linspace(0, 2 * np.pi, self.N_PTS, endpoint=False)
        img_pts = np.array(
            [
                [
                    cx + ra * np.cos(a) * cos_a - rb * np.sin(a) * sin_a,
                    cy + ra * np.cos(a) * sin_a + rb * np.sin(a) * cos_a,
                ]
                for a in angles
            ],
            dtype=np.float32,
        )
        return self._solve(self._obj_pts, img_pts)

    def _from_bbox(self, bbox: tuple) -> tuple:
        x, y, w, h = bbox
        img_pts = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
        return self._solve(self._obj_pts_rect, img_pts)

    def _solve(
        self, obj_pts: np.ndarray, img_pts: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                self._cam_mat,
                self._dist,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if ok and tvec is not None and np.linalg.norm(tvec) < 10.0:
                # Refine with Levenberg-Marquardt
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_pts, img_pts, self._cam_mat, self._dist, rvec, tvec
                )
                return rvec, tvec, True
        except cv2.error:
            pass
        return None, None, False
