"""Standalone scene graph node implementation.

The implementation sticks to numpy and the procedural pyrr API so it can be
vendored without extra engine dependencies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pyrr import matrix44, quaternion


class Object3D:
    """Minimal scene graph node with right-handed transform conventions."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self.position = np.zeros(3, dtype=np.float64)
        self.rotation = quaternion.create(dtype=np.float64)
        self.scale = np.ones(3, dtype=np.float64)

        self.parent: Optional[Object3D] = None
        self.children: list[Object3D] = []

        self.local_transform = matrix44.create_identity(dtype=np.float64)
        self.world_transform = matrix44.create_identity(dtype=np.float64)

        self.update_world_transform()

    # ---------------------------------------------------------------------
    # Hierarchy helpers
    def add_child(self, child: "Object3D") -> None:
        if child is self:
            raise ValueError("an Object3D cannot parent itself")

        if child.parent is self:
            return

        if child.parent is not None:
            child.parent.remove_child(child)

        self.children.append(child)
        child.parent = self
        child.update_world_transform(self.world_transform)

    def remove_child(self, child: "Object3D") -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def traverse(self):
        """Yield this node followed by all descendants (depth-first)."""

        stack: list[Object3D] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(list(reversed(node.children)))

    def get_by_name(self, target: str) -> Optional[Object3D]:
        """Return the first descendant (including self) matching ``target``."""

        for node in self.traverse():
            if node.name == target:
                return node
        return None

    # ------------------------------------------------------------------
    # Rotation helpers (angles in radians)
    def rotate_x(self, angle: float) -> None:
        rot = quaternion.create_from_x_rotation(angle, dtype=self.rotation.dtype)
        self.rotation = quaternion.normalize(quaternion.cross(rot, self.rotation))
        self.update_world_transform(self.parent.world_transform if self.parent else None)

    def rotate_y(self, angle: float) -> None:
        rot = quaternion.create_from_y_rotation(angle, dtype=self.rotation.dtype)
        self.rotation = quaternion.normalize(quaternion.cross(rot, self.rotation))
        self.update_world_transform(self.parent.world_transform if self.parent else None)

    def rotate_z(self, angle: float) -> None:
        rot = quaternion.create_from_z_rotation(angle, dtype=self.rotation.dtype)
        self.rotation = quaternion.normalize(quaternion.cross(rot, self.rotation))
        self.update_world_transform(self.parent.world_transform if self.parent else None)

    # ------------------------------------------------------------------
    # Transform updates
    def update_world_transform(self, parent_world: Optional[np.ndarray] = None) -> None:
        scale_matrix = matrix44.create_from_scale(self.scale, dtype=self.local_transform.dtype)
        rotation_matrix = matrix44.create_from_quaternion(self.rotation, dtype=self.local_transform.dtype)
        translation_matrix = matrix44.create_from_translation(self.position, dtype=self.local_transform.dtype)

        local = matrix44.create_identity(dtype=self.local_transform.dtype)
        local = matrix44.multiply(local, scale_matrix)
        local = matrix44.multiply(local, rotation_matrix)
        local = matrix44.multiply(local, translation_matrix)

        self.local_transform = local

        # TODO it should use local matrix if no parent_world, no ?
        parent_matrix = parent_world if parent_world is not None else matrix44.create_identity(dtype=local.dtype)
        self.world_transform = matrix44.multiply(parent_matrix, local)

        for child in self.children:
            child.update_world_transform(self.world_transform)
