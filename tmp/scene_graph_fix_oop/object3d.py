"""Core scene graph node for the standalone OOP experiment."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from pyrr import Matrix44, Quaternion, Vector3


def _ensure_vec3(value: Iterable[float]) -> Vector3:
    """Return a Vector3 built from the provided iterable."""
    arr = Vector3(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-element iterable for vec3 values")
    return arr


def _quat_identity() -> Quaternion:
    quat = Quaternion(dtype=np.float32)
    quat[3] = 1.0
    return quat


def _quat_from_eulers(euler: Vector3) -> Quaternion:
    return Quaternion.from_eulers(euler, dtype=np.float32)


def _quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return Quaternion(
        [
            ax * bw + aw * bx + ay * bz - az * by,
            ay * bw + aw * by + az * bx - ax * bz,
            az * bw + aw * bz + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _quat_from_axis_rotation(axis: str, angle: float) -> Quaternion:
    axis = axis.lower()
    if axis == "x":
        return Quaternion.from_x_rotation(angle, dtype=np.float32)
    if axis == "y":
        return Quaternion.from_y_rotation(angle, dtype=np.float32)
    if axis == "z":
        return Quaternion.from_z_rotation(angle, dtype=np.float32)
    raise ValueError("Axis must be 'x', 'y', or 'z'")


def _quat_to_eulers(quat: Quaternion) -> Vector3:
    x, y, z, w = quat
    # Yaw (Z), Pitch (Y), Roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return Vector3([roll, pitch, yaw], dtype=np.float32)


class Object3D:
    """Scene node storing transform state and managing a hierarchy."""

    def __init__(self) -> None:
        self.position: Vector3 = Vector3([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation: Quaternion = _quat_identity()
        self.scale: Vector3 = Vector3([1.0, 1.0, 1.0], dtype=np.float32)

        self.parent: Optional["Object3D"] = None
        self.children: List["Object3D"] = []

        self._local_transform: Matrix44 = Matrix44.identity(dtype=np.float32)
        self.world_transform: Matrix44 = Matrix44.identity(dtype=np.float32)
        self._local_dirty: bool = True

    @property
    def local_transform(self) -> Matrix44:
        """Return the cached local transform, recomputing when dirty."""
        if self._local_dirty:
            self._recompute_local_transform()
        return self._local_transform

    def add_child(self, child: "Object3D") -> None:
        if child.parent is not None:
            raise ValueError("Child already has a parent")
        self.children.append(child)
        child.parent = self
        child.update_world_transform()

    def remove_child(self, child: "Object3D") -> None:
        if child not in self.children:
            raise ValueError("Child not found")
        self.children.remove(child)
        child.parent = None
        child.update_world_transform()

    def set_position(self, value: Iterable[float]) -> None:
        self.position = _ensure_vec3(value)
        self._mark_local_dirty()

    def set_scale(self, value: Iterable[float]) -> None:
        self.scale = _ensure_vec3(value)
        self._mark_local_dirty()

    def set_rotation_euler(self, euler: Iterable[float]) -> None:
        euler_vec = _ensure_vec3(euler)
        self.rotation = _quat_from_eulers(euler_vec)
        self._mark_local_dirty()

    def get_rotation_euler(self) -> Vector3:
        return _quat_to_eulers(self.rotation)

    def rotate_x(self, angle: float) -> None:
        delta = _quat_from_axis_rotation("x", angle)
        self.rotation = _quat_multiply(self.rotation, delta)
        self._mark_local_dirty()

    def rotate_y(self, angle: float) -> None:
        delta = _quat_from_axis_rotation("y", angle)
        self.rotation = _quat_multiply(self.rotation, delta)
        self._mark_local_dirty()

    def rotate_z(self, angle: float) -> None:
        delta = _quat_from_axis_rotation("z", angle)
        self.rotation = _quat_multiply(self.rotation, delta)
        self._mark_local_dirty()

    def update_world_transform(self, parent_world: Optional[Matrix44] = None) -> None:
        if self._local_dirty:
            self._recompute_local_transform()

        if parent_world is None and self.parent is not None:
            parent_world = self.parent.world_transform

        if parent_world is None:
            self.world_transform = Matrix44(self._local_transform, dtype=np.float32)
        else:
            self.world_transform = Matrix44(parent_world @ self._local_transform, dtype=np.float32)

        for child in self.children:
            child.update_world_transform(self.world_transform)

    def _mark_local_dirty(self) -> None:
        self._local_dirty = True

    def _recompute_local_transform(self) -> None:
        scale_mat = Matrix44.from_scale(self.scale, dtype=np.float32)
        rot_mat = Matrix44.from_quaternion(self.rotation, dtype=np.float32)
        trans_mat = Matrix44.from_translation(self.position, dtype=np.float32)

        sr = rot_mat @ scale_mat
        self._local_transform = Matrix44(trans_mat @ sr, dtype=np.float32)
        self._local_dirty = False

    def get_world_position(self) -> Vector3:
        world = np.asarray(self.world_transform)
        return Vector3(world[3, :3], dtype=np.float32)
