"""Basic scene graph node implementation."""

from __future__ import annotations

from typing import Generator, List, Optional

import numpy as np
from pyrr import matrix44, quaternion


def _ensure_vec3(value: Optional[np.ndarray] = None) -> np.ndarray:
    """Ensure the input is a 3-component numpy array of floats."""
    if value is None:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Vector3 values must have exactly three components")
    return arr


class Object3D:
    """Scene graph node holding transform state and children."""

    def __init__(
        self,
        name: str = "Object3D",
        position: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        # Name of the object
        self.name = name
        # Local position (3D vector)
        self.position: np.ndarray = _ensure_vec3(position)
        # Local rotation (quaternion)
        self.rotation: np.ndarray = quaternion.create(dtype=float) if rotation is None else quaternion.normalize(np.asarray(rotation, dtype=float))
        # Local scale (3D vector)
        self.scale: np.ndarray = _ensure_vec3(scale) if scale is not None else np.array([1.0, 1.0, 1.0], dtype=float)
        # Parent node in the scene graph
        self.parent: Optional["Object3D"] = None
        # List of child nodes
        self.children: List["Object3D"] = []
        # Local transformation matrix (4x4)
        self.local_transform: np.ndarray = matrix44.create_identity(dtype=float)
        # World transformation matrix (4x4)
        self.world_transform: np.ndarray = matrix44.create_identity(dtype=float)
        # Initialize world transform
        self.update_world_transform()

    # =============================================================================
    # add_child, remove_child
    # =============================================================================

    def add_child(self, child: "Object3D") -> None:
        """Add a child node to this object."""
        if child.parent is not None:
            raise ValueError(f"{child.name} already has a parent")
        child.parent = self
        self.children.append(child)
        # Update child's world transform to reflect new parent
        child.update_world_transform(self.world_transform)

    def remove_child(self, child: "Object3D") -> None:
        """Remove a child node from this object."""
        if child not in self.children:
            raise ValueError(f"{child.name} is not a child of {self.name}")
        self.children.remove(child)
        child.parent = None
        # Update child's world transform to reflect removal from parent
        child.update_world_transform(None)

    # =============================================================================
    # rotation_x, rotation_y, rotation_x functions
    # TODO to rename .rotate_x to show it is not a setter
    # =============================================================================

    def rotate_x(self, angle_rad: float) -> None:
        """Rotate the object around the X axis by the given angle (right-handed, radians)."""
        rot = quaternion.create_from_x_rotation(angle_rad, dtype=float)
        self.rotation = quaternion.normalize(quaternion.cross(self.rotation, rot))
        self.update_world_transform()

    def rotate_y(self, angle_rad: float) -> None:
        """Rotate the object around the Y axis by the given angle (right-handed, radians)."""
        rot = quaternion.create_from_y_rotation(angle_rad, dtype=float)
        self.rotation = quaternion.normalize(quaternion.cross(self.rotation, rot))
        self.update_world_transform()

    def rotate_z(self, angle_rad: float) -> None:
        """Rotate the object around the Z axis by the given angle (right-handed, radians)."""
        rot = quaternion.create_from_z_rotation(angle_rad, dtype=float)
        self.rotation = quaternion.normalize(quaternion.cross(self.rotation, rot))
        self.update_world_transform()

    def get_rotation_euler(self) -> np.ndarray:
        """Return the local rotation as XYZ Euler angles in radians."""
        rot_matrix = matrix44.create_from_quaternion(self.rotation, dtype=float)
        rot_3x3 = rot_matrix[:3, :3]
        sy = np.sqrt(rot_3x3[0, 0] ** 2 + rot_3x3[1, 0] ** 2)
        # pyrr's procedural API lacks quaternion->Euler helpers, so we unwrap manually.
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(rot_3x3[2, 1], rot_3x3[2, 2])
            y = np.arctan2(-rot_3x3[2, 0], sy)
            z = np.arctan2(rot_3x3[1, 0], rot_3x3[0, 0])
        else:
            x = np.arctan2(-rot_3x3[1, 2], rot_3x3[1, 1])
            y = np.arctan2(-rot_3x3[2, 0], sy)
            z = 0.0
        return np.array([x, y, z], dtype=float)

    def rotation_x(self) -> float:
        """Return the local rotation around the X axis in radians."""
        return self.get_rotation_euler()[0]

    def rotation_y(self) -> float:
        """Return the local rotation around the Y axis in radians."""
        return self.get_rotation_euler()[1]

    def rotation_z(self) -> float:
        """Return the local rotation around the Z axis in radians."""
        return self.get_rotation_euler()[2]

    # =============================================================================
    # World Transform
    # =============================================================================

    def update_world_transform(self, parent_world: Optional[np.ndarray] = None) -> None:
        """
        Update the local and world transformation matrices.
        If parent_world is provided, use it as the parent's world transform.
        Transform order: scale -> rotation -> translation (right-handed, row-major)
        """
        # Create scale, rotation, and translation matrices
        scale_matrix = matrix44.create_from_scale(self.scale, dtype=float)
        rotation_matrix = matrix44.create_from_quaternion(self.rotation, dtype=float)
        translation_matrix = matrix44.create_from_translation(self.position, dtype=float)
        # Combine transformations: scale -> rotation -> translation (row-major)
        rs = matrix44.multiply(scale_matrix, rotation_matrix)
        local = matrix44.multiply(rs, translation_matrix)
        self.local_transform = local
        # Determine reference transform (parent's world or identity)
        reference = parent_world
        if reference is None and self.parent is not None:
            reference = self.parent.world_transform
        self.world_transform = matrix44.multiply(reference, self.local_transform) if reference is not None else np.copy(self.local_transform)
        # Recursively update children
        for child in self.children:
            child.update_world_transform(self.world_transform)

    # =============================================================================
    # position, rotation, scale at the world level
    # =============================================================================

    def world_position(self) -> np.ndarray:
        """Return the world position of this object as a 3D vector."""
        origin = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        pos = matrix44.apply_to_vector(self.world_transform, origin)
        return np.array(pos[:3], dtype=float)

    def world_rotation(self) -> np.ndarray:
        """Return the world rotation of this object as a quaternion."""
        if self.parent is None:
            return self.rotation
        else:
            parent_world_rot = self.parent.world_rotation()
            return quaternion.normalize(quaternion.cross(parent_world_rot, self.rotation))

    def world_scale(self) -> np.ndarray:
        """Return the world scale of this object as a 3D vector."""
        if self.parent is None:
            return self.scale
        else:
            parent_world_scale = self.parent.world_scale()
            return self.scale * parent_world_scale

    # =============================================================================
    # Utils
    # =============================================================================

    def traverse(self) -> Generator["Object3D", None, None]:
        """
        Generator to traverse this node and all descendants in the scene graph.
        Yields self first, then recursively yields all children.
        """
        yield self
        for child in self.children:
            yield from child.traverse()

    def get_object_by_name(self, name: str) -> Optional["Object3D"]:
        """Return the first object with the given name found in the scene graph, or None if not found."""
        for node in self.traverse():
            if node.name == name:
                return node
        return None
