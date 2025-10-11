from core.object_3d import Object3D
from pyrr import matrix44
import numpy as np
from cameras.camera_base import CameraBase


class CameraOrthographic(Object3D, CameraBase):
    def __init__(self):
        super().__init__()
        self.left = -1.0
        self.right = 1.0
        self.bottom = -1.0
        self.top = 1.0
        self.near = 0.1
        self.far = 100.0

    def get_projection_matrix(self):
        projection_matrix = matrix44.create_orthogonal_projection(
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
        )
        return projection_matrix

    def get_view_matrix(self):
        """View = inverse(world)"""
        world = self.get_world_matrix()
        view_matrix = matrix44.inverse(world)
        return view_matrix
