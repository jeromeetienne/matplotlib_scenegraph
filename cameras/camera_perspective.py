from core.object_3d import Object3D
from pyrr import matrix44
import numpy as np
from cameras.camera_base import CameraBase


class CameraPerspective(Object3D, CameraBase):
    def __init__(self):
        super().__init__()
        self.fovy = 50.0  # degrees
        self.aspect = 1.0  # width / height
        self.near = 0.001  # near clipping plane
        self.far = 100.0  # far clipping plane

    def get_projection_matrix(self):
        projection_matrix = matrix44.create_perspective_projection(
            self.fovy,
            self.aspect,
            self.near,
            self.far,
            dtype=np.float32,
        )
        return projection_matrix

    def get_view_matrix(self):
        """View = inverse(world)"""
        world_matrix = self.get_world_matrix()
        view_matrix = matrix44.inverse(world_matrix)
        return view_matrix
