# pip imports
from pyrr import vector3, matrix44
from math import atan2
import numpy as np
from core.random import Random


class Object3D:
    def __init__(self) -> None:
        self.uuid = Random.random_uuid()
        self.name = "an object3D"
        self.position = vector3.create(0.0, 0.0, 0.0)
        self.rotation_euler = vector3.create(0.0, 0.0, 0.0)  # Euler XYZ, radians
        self.scale = vector3.create(1.0, 1.0, 1.0)

        self.parent: Object3D | None = None
        self._children: list[Object3D] = []

        self._local_matrix = matrix44.create_identity(dtype=np.float32)
        self._world_matrix = matrix44.create_identity(dtype=np.float32)

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

    # =============================================================================
    # Update matrix
    # =============================================================================
    def update_local_matrix(self) -> None:
        scale_m = matrix44.create_from_scale(self.scale, dtype=np.float32)
        rot_m = matrix44.create_from_eulers(self.rotation_euler, dtype=np.float32)
        trans_m = matrix44.create_from_translation(self.position, dtype=np.float32)

        self._local_matrix = trans_m @ rot_m @ scale_m

    def update_world_matrix(self, parent_world_matrix: np.ndarray | None = None) -> None:
        self.update_local_matrix()

        if parent_world_matrix is not None:
            self._world_matrix = self._local_matrix @ parent_world_matrix
            # self._world_matrix = parent_world_matrix @ self._local_matrix
        else:
            self._world_matrix = self._local_matrix

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
