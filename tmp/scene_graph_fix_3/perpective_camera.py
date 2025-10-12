"""Simple perspective camera built on top of the standalone Object3D."""

from __future__ import annotations

import numpy as np
from pyrr import matrix44

from object3d import Object3D


class PerspectiveCamera(Object3D):
    """Scene node representing a perspective camera."""

    def __init__(self, fov: float = 60.0, aspect_ratio: float = 1.0, near: float = 0.1, far: float = 100.0) -> None:
        super().__init__()
        self.fov = float(fov)
        self.aspect_ratio = float(aspect_ratio)
        self.near = float(near)
        self.far = float(far)

    def get_projection_matrix(self) -> np.ndarray:
        return matrix44.create_perspective_projection_matrix(
            np.deg2rad(self.fov),
            self.aspect_ratio,
            self.near,
            self.far,
            dtype=np.float64,
        )

    def get_view_matrix(self) -> np.ndarray:
        self.update_world_transform(self.parent.world_transform if self.parent else None)
        return matrix44.inverse(self.world_transform)


if __name__ == "__main__":
    camera = PerspectiveCamera(fov=75.0, aspect_ratio=16.0 / 9.0)
    camera.position = np.array([0.0, 0.0, 5.0], dtype=np.float64)
    camera.update_world_transform()
    projection = camera.get_projection_matrix()
    view = camera.get_view_matrix()

    print("Projection matrix:\n", np.round(projection, 3))
    print("View matrix:\n", np.round(view, 3))
