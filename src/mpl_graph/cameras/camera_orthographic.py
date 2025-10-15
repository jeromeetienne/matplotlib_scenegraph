# pip imports
from pyrr import matrix44

# local imports
from .camera import Camera


class CameraOrthographic(Camera):
    def __init__(self):
        super().__init__()
        self.name = f"a {CameraOrthographic.__name__}"
        self.left = -1.0
        self.right = 1.0
        self.bottom = -1.0
        self.top = 1.0
        self.near = 0.1
        self.far = 10.0

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
