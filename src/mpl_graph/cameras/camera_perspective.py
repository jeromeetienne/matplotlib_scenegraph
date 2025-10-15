# pip imports
from pyrr import matrix44
import numpy as np

# local imports
from .camera import Camera


class CameraPerspective(Camera):
    def __init__(self, fovy: float = 50.0, aspect: float = 1.0, near: float = 0.001, far: float = 100.0):
        super().__init__()
        self.name = f"a {CameraPerspective.__name__}"
        self.fovy = fovy  # degrees
        self.aspect = aspect  # width / height
        self.near = near  # near clipping plane
        self.far = far  # far clipping plane

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
