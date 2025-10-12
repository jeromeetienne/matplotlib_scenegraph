"""Camera implementation building on Object3D."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pyrr import matrix44

from object3d import Object3D


class Camera(Object3D):
    """Simple perspective camera."""

    def __init__(
        self,
        name: str = "Camera",
        position: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        fov: float = 60.0,
        aspect_ratio: float = 1.0 / 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        super().__init__(name=name, position=position, rotation=rotation, scale=scale)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

    def get_view_matrix(self) -> np.ndarray:
        return matrix44.inverse(self.world_transform)

    def get_projection_matrix(self) -> np.ndarray:
        return matrix44.create_perspective_projection(
            self.fov,
            self.aspect_ratio,
            self.near,
            self.far,
            dtype=float,
        )
        # return matrix44.create_orthogonal_projection(
        #     -5.0,
        #     5.0,
        #     -5.0,
        #     5.0,
        #     self.near,
        #     self.far,
        #     dtype=float,
        # )
