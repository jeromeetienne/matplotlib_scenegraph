# pip imports
from pyrr import vector3, matrix44
from math import atan2
import numpy as np
from typing import Callable
import math
from typing import Protocol

# local imports
from .random import Random
from .event import Event


# We can define the expected function signature using a Protocol for clarity.
class PreRenderingCallback(Protocol):
    def __call__(self, object_3d: "Object3D", camera: "CameraBase") -> None: ...  # type: ignore


class PostTransformCallback(Protocol):
    def __call__(self, vertices_transformed: np.ndarray) -> None: ...  # type: ignore


class PostRenderingCallback(Protocol):
    def __call__(self) -> None: ...  # type: ignore


class Object3D:
    __slots__ = (
        "uuid",
        "name",
        "position",
        "rotation_euler",
        "scale",
        "parent",
        "_children",
        "_local_matrix",
        "_world_matrix",
        "pre_rendering",
        "post_transform",
        "post_rendering",
    )

    def __init__(self) -> None:
        self.uuid = Random.random_uuid()
        self.name = f"a {Object3D.__name__}"

        self.position = vector3.create(0.0, 0.0, 0.0)
        """Position vector (x, y, z) in local space."""
        self.rotation_euler = vector3.create(0.0, 0.0, 0.0)  # Euler XYZ, radians
        """Rotation euler angles (rx, ry, rz) in local space, in radians."""
        self.scale = vector3.create(1.0, 1.0, 1.0)
        """Scale vector (sx, sy, sz) in local space."""

        self.parent: Object3D | None = None
        """Parent Object3D instance or None."""
        self._children: list[Object3D] = []
        """List of child Object3D instances."""

        self._local_matrix = matrix44.create_identity(dtype=np.float32)
        self._world_matrix = matrix44.create_identity(dtype=np.float32)

        self.pre_rendering = Event[PreRenderingCallback]()
        """Event triggered before rendering the visual."""

        self.post_transform = Event[PostTransformCallback]()
        """
        Event triggered after applying 3d transformations to the visual.

        Arguments sent to subscribers:
        - renderer: The renderer instance performing the rendering.
        - camera: The camera used for rendering.
        - transformed_positions: The numpy array of transformed positions (shape: n x 3).
        """

        self.post_rendering = Event[PostRenderingCallback]()
        """Event triggered after rendering the visual."""

    # =============================================================================
    # add/remove child
    # =============================================================================
    def add_child(self, child: "Object3D") -> None:
        child.parent = self
        self._children.append(child)

    def remove_child(self, child: "Object3D") -> None:
        assert child in self._children, "Child not found"
        self._children.remove(child)
        child.parent = None

    def traverse(self) -> list["Object3D"]:
        objects: list[Object3D] = [self]
        for child in self._children:
            child_objects = child.traverse()
            objects.extend(child_objects)
        return objects

    def root(self) -> "Object3D":
        object: Object3D = self
        while object.parent is not None:
            object = object.parent
        return object

    # =============================================================================
    # Update matrix
    # =============================================================================
    def update_local_matrix(self) -> None:
        scale_matrix = matrix44.create_from_scale(self.scale, dtype=np.float32)
        # REQUIRED: % (math.pi*2) is needed because matrix44.create_from_eulers() doesnt handle angles > 2pi correctly... OOPSSAAA
        # - try: .rotation[2] = time.time() ...
        new_rotation_euler = np.array(
            [self.rotation_euler[0] % (math.pi * 2), self.rotation_euler[2] % (math.pi * 2), self.rotation_euler[1] % (math.pi * 2)], dtype=np.float32
        )
        rotation_matrix = matrix44.create_from_eulers(new_rotation_euler, dtype=np.float32)
        translation_matrix = matrix44.create_from_translation(self.position, dtype=np.float32)

        # self._local_matrix = trans_m @ rot_m @ scale_m
        # self._local_matrix = scale_matrix @ rotation_matrix @ translation_matrix

        # compute the local matrix: first `scale`, then `rotate`, then `translate`
        self._local_matrix = matrix44.create_identity(dtype=np.float32)
        self._local_matrix = matrix44.multiply(self._local_matrix, scale_matrix)
        self._local_matrix = matrix44.multiply(self._local_matrix, rotation_matrix)
        self._local_matrix = matrix44.multiply(self._local_matrix, translation_matrix)

    def update_world_matrix(self, parent_world_matrix: np.ndarray | None = None) -> None:
        self.update_local_matrix()

        if parent_world_matrix is not None:
            # Compute world matrix by combining local and parent world matrices: first `local` then `parent's world`
            self._world_matrix = matrix44.create_identity(dtype=np.float32)
            self._world_matrix = matrix44.multiply(self._world_matrix, self._local_matrix)
            self._world_matrix = matrix44.multiply(self._world_matrix, parent_world_matrix)
        else:
            self._world_matrix = self._local_matrix.copy()

        for child in self._children:
            child.update_world_matrix(self._world_matrix)

    def get_local_matrix(self) -> np.ndarray:
        # self.update_local_matrix()
        return self._local_matrix

    def get_world_matrix(self) -> np.ndarray:
        # self.update_world_matrix()
        return self._world_matrix

    # =============================================================================
    # get_world_position/scale/rotation_euler
    # =============================================================================
    def get_world_position(self) -> np.ndarray:
        return self._world_matrix[3, :3]

    def get_world_scale(self) -> np.ndarray:
        sx = np.linalg.norm(self._world_matrix[0, :3])
        sy = np.linalg.norm(self._world_matrix[1, :3])
        sz = np.linalg.norm(self._world_matrix[2, :3])
        return vector3.create(sx, sy, sz)  # type: ignore

    def get_world_rotation_euler(self) -> np.ndarray:
        # TODO use library to do that... you dont want complex math in your code...
        rot_m = self._world_matrix[:3, :3]
        sy = np.sqrt(rot_m[0, 0] ** 2 + rot_m[1, 0] ** 2)

        singular = sy < 1e-6
        if not singular:
            x = atan2(rot_m[2, 1], rot_m[2, 2])
            y = atan2(-rot_m[2, 0], sy)
            z = atan2(rot_m[1, 0], rot_m[0, 0])
        else:
            x = atan2(-rot_m[1, 2], rot_m[1, 1])
            y = atan2(-rot_m[2, 0], sy)
            z = 0

        return vector3.create(x, y, z)
