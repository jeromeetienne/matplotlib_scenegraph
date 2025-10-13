# stdlib imports
import typing
import math

# pip imports
import matplotlib.pyplot
import matplotlib.backend_bases
import numpy as np
from pyrr import matrix44

# local imports
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_base import CameraBase


class ObjectControllerWasd:
    def __init__(self, renderer: Renderer, object: Object3D) -> None:
        self._renderer = renderer
        self._object = object
        self._speed_translation = 1
        self._speed_rotation = math.pi / 2  # radians per second

        self._vector_translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._vector_rotation = np.array([0.0, 0.0, 0], dtype=np.float32)

        # internal state
        self._mpl_keypress_cid: int | None = None
        """id of keypress event connection"""
        self._mpl_keyrelease_cid: int | None = None
        """id of keyrelease event connection"""
        # remove default matplotlib key bindings that conflict with our controls
        matplotlib.pyplot.rcParams["keymap.save"].remove("s")

    # =============================================================================
    # Public Functions
    # =============================================================================
    def start(self):
        """start controlling the object with the keyboard"""
        # connect the keyboard events
        self._mpl_keypress_cid = self._renderer.get_figure().canvas.mpl_connect("key_press_event", self._on_key_press)
        self._mpl_keyrelease_cid = self._renderer.get_figure().canvas.mpl_connect("key_release_event", self._on_key_release)

    def stop(self):
        """stop controlling the object with the keyboard"""
        # disconnect the keyboard events
        if self._mpl_keypress_cid is not None:
            self._renderer.get_figure().canvas.mpl_disconnect(self._mpl_keypress_cid)
        self._mpl_keypress_cid = None
        if self._mpl_keyrelease_cid is not None:
            self._renderer.get_figure().canvas.mpl_disconnect(self._mpl_keyrelease_cid)
        self._mpl_keyrelease_cid = None

    def update(self, delta_time: float):
        """
        update the object position/rotation based on keyboard state

        Args:
            delta_time (float): time elapsed since last update in seconds
        Returns:
            bool: True if the object has moved, False otherwise
        """
        # print(f"CameraController.update: delta_time={delta_time}, translation={self._vector_translation}, rotation={self._vector_rotation}")

        translation_transformed = self._transform_vector_local_to_world(self._vector_translation)
        # print(f"translation_transformed: {translation_transformed}")
        # print(f"vector_translation: {self._vector_translation}, vector_rotation: {self._vector_rotation}")
        self._object.position += translation_transformed * self._speed_translation * delta_time
        self._object.rotation_euler += self._vector_rotation * self._speed_rotation * delta_time

        has_moved = np.linalg.norm(self._vector_translation) > 0 or np.linalg.norm(self._vector_rotation) > 0
        return has_moved

    # =============================================================================
    # Maintain the rotation/translation vectors
    # =============================================================================

    def _state_change(self, key_str: str, is_pressed: bool):
        # print(f"Key state change: '{key_str}' to {'pressed' if is_pressed else 'released'}")
        if (key_str == "w" or key_str == "up") or (key_str == "shift+w" or key_str == "shift+up"):
            self._vector_translation[2] = -1 * is_pressed
        elif (key_str == "s" or key_str == "down") or (key_str == "shift+s" or key_str == "shift+down"):
            self._vector_translation[2] = +1 * is_pressed
        elif key_str == "shift+a" or key_str == "shift+left":
            self._vector_translation[0] = -1 * is_pressed
        elif key_str == "shift+d" or key_str == "shift+right":
            self._vector_translation[0] = +1 * is_pressed
        elif key_str == "a" or key_str == "left":
            self._vector_rotation[1] = -1 * is_pressed
        elif key_str == "d" or key_str == "right":
            self._vector_rotation[1] = +1 * is_pressed
        else:
            pass

    # =============================================================================
    # event handlers
    # =============================================================================
    def _on_key_press(self, event: matplotlib.backend_bases.Event):
        # get the key event
        keyEvent = typing.cast(matplotlib.backend_bases.KeyEvent, event)
        assert isinstance(keyEvent.key, str), f"keyEvent.key must be str, got {type(keyEvent.key)}"
        # update the keyboard state
        self._state_change(keyEvent.key, True)

    def _on_key_release(self, event: matplotlib.backend_bases.Event):
        # get the key event
        keyEvent = typing.cast(matplotlib.backend_bases.KeyEvent, event)
        assert isinstance(keyEvent.key, str), f"keyEvent.key must be str, got {type(keyEvent.key)}"
        # update the keyboard state
        self._state_change(keyEvent.key, False)

    # =============================================================================
    # transform direction from local space to world space
    # =============================================================================
    def _transform_vector_local_to_world(self, untransformed_direction: np.ndarray) -> np.ndarray:
        """transform a direction vector from local space to world space for the camera"""

        # sanity check
        assert untransformed_direction.shape == (3,)

        # make it homogeneous
        vector_local = np.array(untransformed_direction).copy().astype(np.float32)

        world_matrix = self._object.get_world_matrix()
        vector_homogenous = np.append(vector_local, 0.0)  # make it homogeneous
        vector_homogenous = matrix44.apply_to_vector(world_matrix, vector_local)
        # convert back to 3D
        vector_world = vector_homogenous[0:3]

        vector_world -= self._object.get_world_position()
        # normalize the direction if not zeros
        if np.linalg.norm(vector_world) > 0:
            vector_world /= np.linalg.norm(vector_world)

        return vector_world
