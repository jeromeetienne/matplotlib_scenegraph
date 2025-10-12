# pip imports
import numpy as np

# local imports
from mpl_graph.core.geometry import Geometry


class GeometryShape(Geometry):
    @staticmethod
    def plane(width: float = 1.0, height: float = 1.0) -> Geometry:
        half_width = width / 2.0
        half_height = height / 2.0
        vertices = np.array(
            [
                [-half_width, -half_height, 0.0],
                [half_width, -half_height, 0.0],
                [half_width, half_height, 0.0],
                [-half_width, half_height, 0.0],
            ]
        )
        faces_indices = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        uvs_coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        normals_coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        geometry = Geometry(vertices, faces_indices, uvs_coords, normals_coords)
        return geometry
