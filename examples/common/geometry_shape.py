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

    @staticmethod
    def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0) -> Geometry:
        half_width = width / 2.0
        half_height = height / 2.0
        half_depth = depth / 2.0
        vertices = np.array(
            [
                # front face
                [-half_width, -half_height, half_depth],
                [half_width, -half_height, half_depth],
                [half_width, half_height, half_depth],
                [-half_width, half_height, half_depth],
                # back face
                [-half_width, -half_height, -half_depth],
                [half_width, -half_height, -half_depth],
                [half_width, half_height, -half_depth],
                [-half_width, half_height, -half_depth],
            ]
        )
        faces_indices = np.array(
            [
                # front face
                [0, 1, 2],
                [0, 2, 3],
                # right face
                [1, 5, 6],
                [1, 6, 2],
                # back face
                [5, 4, 7],
                [5, 7, 6],
                # left face
                [4, 0, 3],
                [4, 3, 7],
                # top face
                [3, 2, 6],
                [3, 6, 7],
                # bottom face
                [4, 5, 1],
                [4, 1, 0],
            ]
        )
        uvs_coords = np.array(
            [
                # front face
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                # back face
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        normals_coords = np.array(
            [
                # front face
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        geometry = Geometry(vertices, faces_indices, uvs_coords, normals_coords)
        return geometry
