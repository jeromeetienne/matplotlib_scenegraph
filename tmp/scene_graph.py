"""Minimal scene graph demo using pyrr and matplotlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pyrr import matrix44


def _vec3(value: Iterable[float], default: float = 0.0) -> np.ndarray:
    arr = np.array(list(value) if value is not None else [default, default, default], dtype=float)
    if arr.size != 3:
        raise ValueError("Vector3 expects three components")
    return arr


@dataclass
class Object3D:
    name: str = "Object3D"
    position: np.ndarray = field(default_factory=lambda: _vec3([0.0, 0.0, 0.0]))
    rotation_euler: np.ndarray = field(default_factory=lambda: _vec3([0.0, 0.0, 0.0]))
    scale: np.ndarray = field(default_factory=lambda: _vec3([1.0, 1.0, 1.0]))
    parent: Optional["Object3D"] = field(default=None, init=False, repr=False)
    children: List["Object3D"] = field(default_factory=list, init=False, repr=False)

    def add_child(self, child: "Object3D") -> None:
        if child.parent is not None:
            raise ValueError(f"{child.name} already has a parent")
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "Object3D") -> None:
        if child not in self.children:
            raise ValueError(f"{child.name} is not a child of {self.name}")
        self.children.remove(child)
        child.parent = None

    def local_matrix(self) -> np.ndarray:
        scale_m = matrix44.create_from_scale(self.scale)
        rot_m = matrix44.create_from_eulers(self.rotation_euler)
        trans_m = matrix44.create_from_translation(self.position)
        rs = matrix44.multiply(rot_m, scale_m)
        return matrix44.multiply(trans_m, rs)

    def world_matrix(self) -> np.ndarray:
        local = self.local_matrix()
        if self.parent is None:
            return local
        return matrix44.multiply(self.parent.world_matrix(), local)

    def world_position(self) -> np.ndarray:
        wm = self.world_matrix()
        return np.array(wm[:3, 3])

    def traverse(self) -> Generator["Object3D", None, None]:
        yield self
        for child in self.children:
            yield from child.traverse()


@dataclass
class PerspectiveCamera(Object3D):
    fov_y_degrees: float = 60.0
    aspect: float = 16.0 / 9.0
    z_near: float = 0.1
    z_far: float = 100.0

    def projection_matrix(self) -> np.ndarray:
        return matrix44.create_perspective_projection(
            self.fov_y_degrees,
            self.aspect,
            self.z_near,
            self.z_far,
        )

    def view_matrix(self) -> np.ndarray:
        return matrix44.inverse(self.world_matrix())


def render_scene_matplotlib(root: Object3D, camera: PerspectiveCamera, ax: Optional[Axes] = None) -> Axes:
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)

    projection = camera.projection_matrix()
    view = camera.view_matrix()

    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    for node in root.traverse():
        if node is camera:
            continue
        world_pos = node.world_position()
        view_pos = view @ np.array([*world_pos, 1.0])
        clip = projection @ view_pos
        if clip[3] <= 0:
            continue
        ndc = clip[:3] / clip[3]
        if np.any(np.abs(ndc[:2]) > 1.0):
            continue
        xs.append(ndc[0])
        ys.append(ndc[1])
        labels.append(node.name)

    ax.clear()
    ax.scatter(xs, ys, c="tab:blue")
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, fontsize=8, ha="center")
    ax.set_xlabel("X (NDC)")
    ax.set_ylabel("Y (NDC)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Scene Graph Projection")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal")
    return ax


def _build_demo_scene() -> Tuple[Object3D, PerspectiveCamera]:
    root = Object3D(name="World")
    camera = PerspectiveCamera(name="Camera")
    camera.position = _vec3([0.0, 0.0, 5.0])
    root.add_child(camera)

    cube = Object3D(name="Cube")
    cube.position = _vec3([-1.0, 0.0, 0.0])
    cube.rotation_euler = _vec3([0.0, np.pi / 4, 0.0])
    root.add_child(cube)

    sphere = Object3D(name="Sphere")
    sphere.position = _vec3([1.0, 0.5, -0.5])
    sphere.scale = _vec3([0.5, 0.5, 0.5])
    cube.add_child(sphere)

    return root, camera


if __name__ == "__main__":
    scene_root, camera = _build_demo_scene()
    ax = render_scene_matplotlib(scene_root, camera)
    plt.show()
