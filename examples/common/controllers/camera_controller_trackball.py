"""
Trackball camera controller for Matplotlib scenes.

This controller orbits a camera around a target point, pans the target in world
space using the camera's axes, and dollies (zooms) by moving the camera closer
to or further from the target. It integrates with the AnimationLoop pattern by
exposing update(), which returns True when the camera changed so the scene can
be re-rendered.

Usage with AnimationLoop:

    controller = CameraControllerTrackball(renderer, camera, target=np.array([0,0,0], dtype=np.float32))
    controller.start()

    @animation_loop.decorator_callback
    def update_camera(delta_time: float):
    moved = controller.update(delta_time)
    return scene.traverse() if moved else []

Controls (defaults):
- Left mouse drag: orbit around target (azimuth/elevation-like)
- Middle mouse drag or Shift+Left: pan (truck/pedestal) in world space
- Right mouse drag or Ctrl+Left: dolly (move forward/back)
- Mouse wheel/trackpad scroll: dolly zoom

Notes:
- Internally stores a target point and an offset vector camera.position - target.
- Rotations use pyrr quaternions (no manual trig math); look_at keeps orientation.
- Panning uses camera's world right and up axes extracted from its world matrix.
"""

# stdlib imports
from typing import Optional, Any
import math

# pip imports
import numpy as np

# Matplotlib event base types; use concrete MouseEvent where helpful
import matplotlib.backend_bases
from pyrr import quaternion

# local imports
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.cameras.camera_base import CameraBase
from mpl_graph.cameras.camera_perspective import CameraPerspective


class CameraControllerTrackball:
    def __init__(
        self,
        renderer: Renderer,
        camera: CameraBase,
        target: Optional[np.ndarray] = None,
        *,
        invert_orbit_x: bool = True,
        invert_orbit_y: bool = False,
        invert_pan_x: bool = False,
        invert_pan_y: bool = True,
        invert_dolly_drag: bool = True,
        invert_scroll: bool = False,
        max_polar_angle_deg: float = 89.5,
    ) -> None:
        """
        A trackball-style camera controller for orbiting/panning/dollying a camera.
        """
        self._renderer = renderer
        self._camera = camera

        # Internal orbital state (around target) as an offset vector.
        # We maintain camera.position = target + offset at all times.
        cam_pos = np.array(self._camera.position, dtype=np.float32)
        tgt = np.array(target if target is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self._target = tgt.astype(np.float32)
        self._offset = (cam_pos - self._target).astype(np.float32)
        if float(np.linalg.norm(self._offset)) < 1e-6:
            self._offset[:] = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self._distance = float(np.linalg.norm(self._offset))

        # event connections
        self._mpl_press_cid: int | None = None
        self._mpl_release_cid: int | None = None
        self._mpl_motion_cid: int | None = None
        self._mpl_scroll_cid: int | None = None

        # Interaction state (for dragging and buttons)
        self._is_dragging = False
        self._last_x = 0.0
        self._last_y = 0.0
        self._active_button: int | None = None  # 1=left,2=middle,3=right

        # sensitivity/limits
        self.rotate_speed = 0.005  # radians per pixel
        self.pan_speed = 0.001  # world units per pixel per unit distance
        self.dolly_speed = 0.002  # distance delta per pixel
        self.scroll_zoom_factor = 1.1  # multiplicative factor per scroll step
        self.min_distance = 0.05
        self.max_distance = 1e6
        self.max_polar_angle = math.radians(float(max_polar_angle_deg))

        # Direction tuning flags
        # Defaults chosen for typical screen coordinates (y grows downward) and natural feel:
        # - orbit yaw feels natural with invert_orbit_x=True (left drag rotates scene left)
        # - orbit pitch feels natural with invert_orbit_y=False when dy<0 for upward drag
        self.invert_orbit_x = bool(True if invert_orbit_x is None else invert_orbit_x)
        self.invert_orbit_y = bool(False if invert_orbit_y is None else invert_orbit_y)
        self.invert_pan_x = bool(invert_pan_x)
        self.invert_pan_y = bool(invert_pan_y)
        self.invert_dolly_drag = bool(invert_dolly_drag)
        self.invert_scroll = bool(invert_scroll)

        # Dirty flag to integrate with AnimationLoop.update
        self._dirty = True  # ensure one initial update to push exact state
        self._apply_state()  # ensure camera is aligned to state

    # =============================================================================
    # Public API
    # =============================================================================
    def start(self) -> None:
        """Connect matplotlib mouse events to enable interaction."""
        fig = self._renderer.get_figure()
        canvas = fig.canvas
        self._mpl_press_cid = canvas.mpl_connect("button_press_event", self._on_button_press)
        self._mpl_release_cid = canvas.mpl_connect("button_release_event", self._on_button_release)
        self._mpl_motion_cid = canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._mpl_scroll_cid = canvas.mpl_connect("scroll_event", self._on_scroll)

    def stop(self) -> None:
        """Disconnect matplotlib events and reset drag state."""
        fig = self._renderer.get_figure()
        canvas = fig.canvas
        if self._mpl_press_cid is not None:
            canvas.mpl_disconnect(self._mpl_press_cid)
        if self._mpl_release_cid is not None:
            canvas.mpl_disconnect(self._mpl_release_cid)
        if self._mpl_motion_cid is not None:
            canvas.mpl_disconnect(self._mpl_motion_cid)
        if self._mpl_scroll_cid is not None:
            canvas.mpl_disconnect(self._mpl_scroll_cid)
        self._mpl_press_cid = self._mpl_release_cid = self._mpl_motion_cid = self._mpl_scroll_cid = None

        self._is_dragging = False
        self._active_button = None

    def update(self, _delta_time: float) -> bool:
        """Call each frame; returns True if camera changed to trigger re-render."""
        has_moved = self._dirty
        self._dirty = False
        return has_moved

    def set_target(self, target: np.ndarray) -> None:
        self._target = np.array(target, dtype=np.float32)
        # recompute offset from new target keeping camera where it is
        self._offset = (np.array(self._camera.position, dtype=np.float32) - self._target).astype(np.float32)
        self._distance = float(np.linalg.norm(self._offset))
        self._distance = max(self.min_distance, min(self.max_distance, self._distance))
        self._apply_state()
        self._dirty = True

    # =============================================================================
    # Event handlers
    # =============================================================================
    def _on_button_press(self, event: Any) -> None:
        # Ignore clicks outside axes; avoids starting drags on figure background.
        if event.inaxes is None:
            return
        self._is_dragging = True
        self._last_x = float(getattr(event, "x", 0.0) or 0.0)
        self._last_y = float(getattr(event, "y", 0.0) or 0.0)
        self._active_button = int(getattr(event, "button", 1) or 1)

    def _on_button_release(self, event: Any) -> None:
        """End a drag sequence; also guard against releases outside the axes."""
        if event.inaxes is None:
            # allow releasing outside
            self._is_dragging = False
            self._active_button = None
            return
        self._is_dragging = False
        self._active_button = None

    def _on_mouse_move(self, event: Any) -> None:
        """During a drag, interpret movement as orbit/pan/dolly based on button/modifiers."""
        if not self._is_dragging:
            return
        if self._active_button is None:
            return
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return

        x = float(getattr(event, "x", 0.0) or 0.0)
        y = float(getattr(event, "y", 0.0) or 0.0)
        dx, dy = x - self._last_x, y - self._last_y
        self._last_x, self._last_y = x, y

        # modifier keys (string like 'shift', 'control', 'alt', or combinations)
        key = getattr(event, "key", None) or ""
        is_shift = isinstance(key, str) and ("shift" in key)
        is_ctrl = isinstance(key, str) and ("control" in key or "cmd" in key or "meta" in key)

        # Map buttons/modifiers to actions
        if self._active_button == 1 and not is_shift and not is_ctrl:
            # orbit
            self._orbit(dx, dy)
        elif (self._active_button == 2) or (self._active_button == 1 and is_shift):
            # pan
            self._pan(dx, dy)
        elif (self._active_button == 3) or (self._active_button == 1 and is_ctrl):
            # dolly
            self._dolly_pixels((-dy) if self.invert_dolly_drag else dy)  # drag up to move forward (zoom in)

    def _on_scroll(self, event: Any) -> None:
        """Zoom in/out by scaling the camera offset based on scroll direction/step."""
        if event.inaxes is None:
            return
        # Matplotlib provides step (int) and button ('up'/'down') in some backends
        step = float(getattr(event, "step", 0.0) or (1.0 if getattr(event, "button", "up") == "up" else -1.0))
        if step != 0:
            factor = self.scroll_zoom_factor ** abs(step)
            # Allow inversion of scroll zoom direction if desired
            effective_step = -step if self.invert_scroll else step
            if effective_step > 0:
                # zoom in: reduce offset length
                self._offset = (self._offset / factor).astype(np.float32)
            else:
                # zoom out: increase offset length
                self._offset = (self._offset * factor).astype(np.float32)
            # clamp
            d = float(np.linalg.norm(self._offset))
            d = max(self.min_distance, min(self.max_distance, d))
            if d > 1e-8:
                self._offset = (self._offset / np.linalg.norm(self._offset) * d).astype(np.float32)
            self._apply_state()
        self._dirty = True

    # =============================================================================
    # Actions
    # =============================================================================
    def _orbit(self, dx: float, dy: float) -> None:
        """Rotate around target using spherical angles with elevation clamp.

        This keeps behavior stable near poles and allows configurable direction inversions.
        """
        # Convert current offset to spherical
        r, theta, phi = self._offset_to_spherical(self._offset)
        # Apply deltas with configurable inversion
        s_yaw = -1.0 if self.invert_orbit_x else 1.0
        s_pitch = 1.0 if not self.invert_orbit_y else -1.0
        theta += s_yaw * float(dx) * self.rotate_speed
        phi += s_pitch * float(dy) * self.rotate_speed
        # Clamp elevation to avoid pole singularities
        phi = max(-self.max_polar_angle, min(self.max_polar_angle, phi))
        # Rebuild offset and apply
        self._offset = self._spherical_to_offset(r, theta, phi)
        self._apply_state()
        self._dirty = True

    def _pan(self, dx: float, dy: float) -> None:
        """Pan target in world space along camera right/up; scale with distance/FOV."""
        # Scale pan with distance and FOV so it feels consistent
        distance = max(self._distance, self.min_distance)
        if isinstance(self._camera, CameraPerspective):
            fovy = math.radians(float(getattr(self._camera, "fovy", 50.0)))
            # pixel -> world scale approximation at target distance
            # world_units_per_pixel vertically ~ 2 * distance * tan(fovy/2) / viewport_height
            height_px = float(self._renderer.get_figure().canvas.get_width_height()[1])
            height_px = max(height_px, 1.0)
            world_per_px_y = 2.0 * distance * math.tan(0.5 * fovy) / height_px
            world_per_px_x = world_per_px_y  # assume square pixels; aspect cancels for screen space
        else:
            # Orthographic: use a simple linear mapping
            world_per_px_x = world_per_px_y = self.pan_speed * distance

        # Use robust axes based on offset rather than world matrix to avoid pole issues
        right, up, _forward = self._orbit_axes()
        # Apply inversion to match user preference/OS settings
        sx = -1.0 if self.invert_pan_x else 1.0
        sy = -1.0 if self.invert_pan_y else 1.0
        pan_world = (sx * dx * world_per_px_x) * right + (sy * dy * world_per_px_y) * up

        self._target = (self._target + pan_world).astype(np.float32)
        # keep camera at same relative offset from target
        self._apply_state()
        self._dirty = True

    def _dolly_pixels(self, dy: float) -> None:
        """Zoom by scaling offset with drag distance; positive dy moves forward."""
        # Scale the offset vector to move toward/away from target
        scale = float(1.0 - self.dolly_speed * dy)
        scale = max(0.01, scale)
        self._offset = (self._offset * scale).astype(np.float32)
        self._distance = float(np.linalg.norm(self._offset))
        self._distance = max(self.min_distance, min(self.max_distance, self._distance))
        # normalize then rescale to clamped distance
        if self._distance > 1e-8:
            self._offset = (self._offset / self._distance * self._distance).astype(np.float32)
        self._apply_state()
        self._dirty = True

    # =============================================================================
    # Helpers
    # =============================================================================
    def _apply_state(self) -> None:
        """Apply target+offset to camera position and orient toward target."""
        # Position camera from target + offset and look at target
        self._camera.position[:] = (self._target + self._offset).astype(np.float32)
        self._camera.look_at(self._target)
        # keep cached distance in sync
        self._distance = float(np.linalg.norm(self._offset))

    def _camera_axes_world(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract camera local axes (right, up, forward) from world matrix, normalized."""
        m = self._camera.get_world_matrix().astype(np.float32)
        # Rows 0,1,2 are the local axes in world space (see Object3D implementation)
        x_axis = m[0, :3]
        y_axis = m[1, :3]
        z_axis = m[2, :3]

        # Normalize to ensure unit vectors (remove scale)
        def _norm(v: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            return (v / n) if n > 1e-12 else v

        x_axis = _norm(x_axis)
        y_axis = _norm(y_axis)
        z_axis = _norm(z_axis)
        return x_axis, y_axis, z_axis

    def _orbit_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute stable right/up/forward axes from target-offset, with fallback near poles.

        forward points from camera toward target; right is world_up x forward. If forward
        is near-colinear with world_up, use an alternate up axis to avoid degeneracy.
        """
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward = (-self._offset).astype(np.float32)
        f_norm = float(np.linalg.norm(forward))
        if f_norm < 1e-12:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = (forward / f_norm).astype(np.float32)

        # candidate right = up x forward
        right = np.cross(world_up, forward).astype(np.float32)
        r_norm = float(np.linalg.norm(right))
        if r_norm < 1e-6:
            # forward ~ parallel to world_up; pick an alternate up axis (Z) to define right
            alt_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(alt_up, forward).astype(np.float32)
            r_norm = float(np.linalg.norm(right))
            if r_norm < 1e-6:
                # As last resort pick X axis
                alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                right = np.cross(alt_up, forward).astype(np.float32)
                r_norm = float(np.linalg.norm(right))
        if r_norm > 0:
            right = (right / r_norm).astype(np.float32)

        up = np.cross(forward, right).astype(np.float32)
        u_norm = float(np.linalg.norm(up))
        if u_norm > 0:
            up = (up / u_norm).astype(np.float32)
        return right, up, forward

    def _rotate_vector_by_quaternion(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion using pyrr helper."""
        res = quaternion.apply_to_vector(q.astype(np.float32), v.astype(np.float32))
        return np.asarray(res, dtype=np.float32)

    # Spherical helpers -------------------------------------------------------
    def _offset_to_spherical(self, offset: np.ndarray) -> tuple[float, float, float]:
        """Return (r, theta, phi) from offset vector.

        theta: azimuth around Y (world up), 0 aligned with +Z
        phi: elevation from horizon ([-pi/2, +pi/2])
        """
        x, y, z = float(offset[0]), float(offset[1]), float(offset[2])
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-12:
            return 0.0, 0.0, 0.0
        theta = math.atan2(x, z if abs(z) > 1e-12 else 1e-12)
        hyp = math.sqrt(max(x * x + z * z, 1e-24))
        phi = math.atan2(y, hyp)
        return r, theta, phi

    def _spherical_to_offset(self, r: float, theta: float, phi: float) -> np.ndarray:
        """Return offset vector from (r, theta, phi) in right-handed coordinates.

        x = r * sin(theta) * cos(phi)
        y = r * sin(phi)
        z = r * cos(theta) * cos(phi)
        """
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        return np.array([r * sin_theta * cos_phi, r * sin_phi, r * cos_theta * cos_phi], dtype=np.float32)
