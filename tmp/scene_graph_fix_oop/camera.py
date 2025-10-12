"""Camera implementation for the standalone OOP scene graph."""

from __future__ import annotations

import numpy as np
from pyrr import Matrix44

from .object3d import Object3D


class Camera(Object3D):
    """Perspective camera with configurable frustum."""

    def __init__(
        self,
        fov: float = 60.0,
        aspect_ratio: float = 16.0 / 9.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        super().__init__()
        self.fov = float(fov)
        self.aspect_ratio = float(aspect_ratio)
        self.near = float(near)
        self.far = float(far)

    def get_view_matrix(self) -> np.ndarray:
        self.update_world_transform()
        view = Matrix44(self.world_transform, dtype=np.float32).inverse
        return np.asarray(view, dtype=np.float32)

    def get_projection_matrix(self) -> np.ndarray:
        projection = Matrix44.perspective_projection(
            self.fov,
            self.aspect_ratio,
            self.near,
            self.far,
            dtype=np.float32,
        )
        return np.asarray(projection, dtype=np.float32)
