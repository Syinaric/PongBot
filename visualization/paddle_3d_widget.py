"""
visualization/paddle_3d_widget.py – POV-locked 3-D paddle renderer.

POV-locked means the virtual camera exactly matches the real webcam:
the OpenGL projection matrix is built from the camera intrinsics so
the paddle appears at the correct angle and relative position in 3-D space.
As the real paddle moves left / right / up / down / closer / further,
the virtual paddle does the same — you are literally watching a 3-D
reconstruction of the camera scene.

Coordinate system conversion
-----------------------------
  OpenCV (solvePnP output):   X right, Y down, Z into scene
  OpenGL display convention:  X right, Y up,   Z toward viewer

Flip matrix F = diag(1, -1, -1) converts between them:
    R_gl = F @ R_cv @ F
    t_gl = F @ t_cv           → (x, -y, -z)

The OpenGL projection matrix is derived from the camera intrinsic matrix K:
    | 2fx/W   0      (W-2cx)/W    0               |
    |  0     2fy/H  -(H-2cy)/H   0               |
    |  0      0     -(f+n)/(f-n)  -2fn/(f-n)     |
    |  0      0     -1            0               |

When tracking is lost the widget falls back to a free-orbit view so the
user can still inspect the paddle model manually.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (
    GL_AMBIENT, GL_AMBIENT_AND_DIFFUSE, GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_DIFFUSE,
    GL_FRONT_AND_BACK, GL_LIGHT0, GL_LIGHTING, GL_LINES, GL_MODELVIEW,
    GL_POSITION, GL_PROJECTION, GL_QUAD_STRIP, GL_QUADS, GL_SMOOTH,
    GL_TRIANGLE_FAN,
    glBegin, glClear, glClearColor, glColor3f, glColorMaterial,
    glDisable, glEnable, glEnd, glLightfv, glLineWidth, glLoadIdentity,
    glLoadMatrixf, glMatrixMode, glMultMatrixf, glNormal3f,
    glPopMatrix, glPushMatrix, glRotatef, glScalef, glShadeModel,
    glTranslatef, glVertex3f, glViewport,
)
from OpenGL.GLU import gluLookAt

from config import Config

_F = np.diag([1.0, -1.0, -1.0])  # OpenCV → OpenGL axis flip


class Paddle3DWidget(QOpenGLWidget):
    """
    3-D OpenGL widget – POV-locked to the real camera.

    When tracking:
      - The projection exactly matches the webcam FOV.
      - The paddle is rendered at its solvePnP position and orientation.
      - Moving the real paddle left/right/closer shows the same in 3-D.

    When not tracking:
      - Falls back to a free-orbit preview (drag with left mouse button).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(500, 480)
        self.setMouseTracking(True)

        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None
        self._visible_side: str = "red"
        self._is_tracking: bool = False

        # Free-orbit state (used when not tracking)
        self._orbit_x: float = 20.0
        self._orbit_y: float = -30.0
        self._last_mouse: tuple[float, float] | None = None

        # Viewport size (updated in resizeGL)
        self._vp_w: int = 500
        self._vp_h: int = 480

    # ── QOpenGLWidget overrides ───────────────────────────────────────────────

    def initializeGL(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 4.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.30, 0.30, 0.30, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.85, 0.85, 0.85, 1.0])
        glClearColor(0.10, 0.10, 0.14, 1.0)

    def resizeGL(self, w: int, h: int) -> None:
        self._vp_w, self._vp_h = w, h
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self._is_tracking and self._rvec is not None and self._tvec is not None:
            self._paint_pov_locked()
        else:
            self._paint_freeorbit()

    # ── POV-locked rendering ──────────────────────────────────────────────────

    def _paint_pov_locked(self) -> None:
        """Render using camera intrinsics → exact-match projection."""
        # ── Projection from intrinsics ────────────────────────────────────────
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(self._intrinsics_projection().T)  # column-major

        # ── View: camera at origin, looking down -Z (OpenGL) ─────────────────
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Camera looks toward -Z in OpenGL; paddle has z_gl = -z_cv < 0
        # No extra gluLookAt needed since we bake the flip into the model matrix.

        self._draw_pov_grid()

        # ── Model: apply pose from solvePnP ──────────────────────────────────
        glPushMatrix()
        glMultMatrixf(self._pose_matrix().T)  # column-major
        self._draw_paddle()
        glPopMatrix()

    def _intrinsics_projection(self) -> np.ndarray:
        """
        Build a 4×4 OpenGL projection matrix from the camera's K matrix.
        Reference: https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
        """
        fx = Config.FOCAL_LENGTH_X
        fy = Config.FOCAL_LENGTH_Y
        cx = Config.PRINCIPAL_POINT_X
        cy = Config.PRINCIPAL_POINT_Y
        W  = float(Config.FRAME_WIDTH)
        H  = float(Config.FRAME_HEIGHT)
        n  = 0.01   # near plane (metres)
        f  = 10.0   # far plane  (metres)

        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] =  2.0 * fx / W
        P[1, 1] =  2.0 * fy / H
        P[0, 2] =  (W - 2.0 * cx) / W
        P[1, 2] = -(H - 2.0 * cy) / H   # flip Y from image to GL coords
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -2.0 * f * n / (f - n)
        P[3, 2] = -1.0
        return P

    def _pose_matrix(self) -> np.ndarray:
        """
        Build a 4×4 model matrix from solvePnP rvec/tvec,
        converting from OpenCV to OpenGL coordinate convention.
        """
        R_cv, _ = cv2.Rodrigues(self._rvec)
        t_cv    = self._tvec.flatten()

        R_gl = _F @ R_cv @ _F
        t_gl = _F @ t_cv          # (x, -y, -z)

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R_gl
        M[:3,  3] = t_gl
        return M

    # ── Free-orbit rendering (no tracking) ───────────────────────────────────

    def _paint_freeorbit(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        from OpenGL.GLU import gluPerspective
        gluPerspective(45.0, self._vp_w / max(self._vp_h, 1), 0.01, 10.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        self._draw_orbit_grid()

        glPushMatrix()
        glRotatef(self._orbit_x, 1.0, 0.0, 0.0)
        glRotatef(self._orbit_y, 0.0, 1.0, 0.0)
        self._draw_paddle()
        glPopMatrix()

    # ── Grid helpers ──────────────────────────────────────────────────────────

    def _draw_pov_grid(self) -> None:
        """Faint XY-plane grid at z = -0.5 m to give depth reference."""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.22, 0.22, 0.28)
        glBegin(GL_LINES)
        step = 0.1
        for i in range(-8, 9):
            v = i * step
            glVertex3f(v, -0.4, -0.5); glVertex3f(v,  0.4, -0.5)
            glVertex3f(-0.8, v, -0.5); glVertex3f( 0.8, v, -0.5)
        glEnd()

        # World origin axes
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(0.9, 0.2, 0.2); glVertex3f(0,0,-0.5); glVertex3f(0.15,0,-0.5)  # X
        glColor3f(0.2, 0.9, 0.2); glVertex3f(0,0,-0.5); glVertex3f(0,0.15,-0.5)  # Y
        glColor3f(0.3, 0.5, 1.0); glVertex3f(0,0,-0.5); glVertex3f(0,0,-0.35)    # Z
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_orbit_grid(self) -> None:
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.25, 0.25, 0.32)
        glBegin(GL_LINES)
        for i in range(-5, 6):
            v = i * 0.1
            glVertex3f(v, -0.5, -0.15); glVertex3f(v,  0.5, -0.15)
            glVertex3f(-0.5, v, -0.15); glVertex3f( 0.5, v, -0.15)
        glEnd()
        glLineWidth(2.5)
        glBegin(GL_LINES)
        glColor3f(1, 0.2, 0.2); glVertex3f(0,0,0); glVertex3f(0.18,0,0)
        glColor3f(0.2, 1, 0.2); glVertex3f(0,0,0); glVertex3f(0,0.18,0)
        glColor3f(0.3,0.5,1.0); glVertex3f(0,0,0); glVertex3f(0,0,0.18)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    # ── Paddle geometry ───────────────────────────────────────────────────────

    def _draw_paddle(self) -> None:
        hw  = Config.PADDLE_HEAD_WIDTH  / 2.0
        hh  = Config.PADDLE_HEAD_HEIGHT / 2.0
        ht  = Config.PADDLE_THICKNESS   / 2.0
        hl  = Config.PADDLE_HANDLE_LENGTH
        hw2 = Config.PADDLE_HANDLE_WIDTH / 2.0

        if self._visible_side == "red":
            front_rgb = (0.82, 0.10, 0.10)
            back_rgb  = (0.07, 0.07, 0.07)
        else:
            front_rgb = (0.07, 0.07, 0.07)
            back_rgb  = (0.82, 0.10, 0.10)

        N = 72

        glColor3f(*front_rgb)
        _ellipse_face(hw, hh, +ht, N, +1)

        glColor3f(*back_rgb)
        _ellipse_face(hw, hh, -ht, N, -1)

        glColor3f(0.62, 0.48, 0.25)
        _ellipse_edge(hw, hh, ht, N)

        glColor3f(0.55, 0.40, 0.20)
        _box(hw2, ht, -hh, -hh - hl)

    # ── Mouse orbit (free mode only) ──────────────────────────────────────────

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = (ev.position().x(), ev.position().y())

    def mouseMoveEvent(self, ev) -> None:
        if self._last_mouse and not self._is_tracking:
            x, y = ev.position().x(), ev.position().y()
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self._orbit_y += dx * 0.5
            self._orbit_x  = max(-89, min(89, self._orbit_x + dy * 0.5))
            self._last_mouse = (x, y)
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        self._last_mouse = None

    # ── Called from main loop ─────────────────────────────────────────────────

    def update_pose(self, state: dict | None) -> None:
        if state:
            self._rvec         = state.get("rvec")
            self._tvec         = state.get("tvec")
            self._visible_side = state.get("visible_side", "red")
            self._is_tracking  = bool(state.get("is_tracking"))
        else:
            self._rvec = self._tvec = None
            self._is_tracking = False
        self.update()


# ── Module-level geometry helpers ────────────────────────────────────────────

def _ellipse_face(rx: float, ry: float, z: float, segs: int, nz: float) -> None:
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0.0, 0.0, nz)
    glVertex3f(0.0, 0.0, z)
    for i in range(segs + 1):
        a = 2.0 * math.pi * i / segs
        glVertex3f(rx * math.cos(a), ry * math.sin(a), z)
    glEnd()


def _ellipse_edge(rx: float, ry: float, ht: float, segs: int) -> None:
    glBegin(GL_QUAD_STRIP)
    for i in range(segs + 1):
        a = 2.0 * math.pi * i / segs
        c, s = math.cos(a), math.sin(a)
        nx, ny = c / rx, s / ry
        ln = math.hypot(nx, ny) or 1.0
        glNormal3f(nx / ln, ny / ln, 0.0)
        glVertex3f(rx * c, ry * s, +ht)
        glVertex3f(rx * c, ry * s, -ht)
    glEnd()


def _box(hx: float, hz: float, y_top: float, y_bot: float) -> None:
    faces = [
        (( 0,  0,  1), [(-hx,y_top, hz),( hx,y_top, hz),( hx,y_bot, hz),(-hx,y_bot, hz)]),
        (( 0,  0, -1), [(-hx,y_top,-hz),(-hx,y_bot,-hz),( hx,y_bot,-hz),( hx,y_top,-hz)]),
        ((-1,  0,  0), [(-hx,y_top, hz),(-hx,y_bot, hz),(-hx,y_bot,-hz),(-hx,y_top,-hz)]),
        (( 1,  0,  0), [( hx,y_top, hz),( hx,y_top,-hz),( hx,y_bot,-hz),( hx,y_bot, hz)]),
        (( 0,  1,  0), [(-hx,y_top, hz),(-hx,y_top,-hz),( hx,y_top,-hz),( hx,y_top, hz)]),
        (( 0, -1,  0), [(-hx,y_bot, hz),( hx,y_bot, hz),( hx,y_bot,-hz),(-hx,y_bot,-hz)]),
    ]
    for (nx, ny, nz), verts in faces:
        glBegin(GL_QUADS)
        glNormal3f(nx, ny, nz)
        for vx, vy, vz in verts:
            glVertex3f(vx, vy, vz)
        glEnd()
