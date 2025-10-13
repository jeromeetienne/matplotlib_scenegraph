# pip imports
import numpy as np

# local imports
from mpl_graph.geometry.geometry import Geometry


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
        half_w = width / 2.0
        half_h = height / 2.0
        half_d = depth / 2.0

        # 8 vertices of the box
        vertices = np.array(
            [
                [-half_w, -half_h, -half_d],
                [half_w, -half_h, -half_d],
                [half_w, half_h, -half_d],
                [-half_w, half_h, -half_d],
                [-half_w, -half_h, half_d],
                [half_w, -half_h, half_d],
                [half_w, half_h, half_d],
                [-half_w, half_h, half_d],
            ]
        )

        # 12 triangles (2 per face)
        faces_indices = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # back
                [4, 6, 5],
                [4, 7, 6],  # front
                [0, 4, 5],
                [0, 5, 1],  # bottom
                [3, 2, 6],
                [3, 6, 7],  # top
                [1, 5, 6],
                [1, 6, 2],  # right
                [0, 3, 7],
                [0, 7, 4],  # left
            ]
        )

        # UVs per vertex (simple cube mapping)
        uvs_coords = np.array(
            [
                [0.0, 0.0],  # 0
                [1.0, 0.0],  # 1
                [1.0, 1.0],  # 2
                [0.0, 1.0],  # 3
                [0.0, 0.0],  # 4
                [1.0, 0.0],  # 5
                [1.0, 1.0],  # 6
                [0.0, 1.0],  # 7
            ]
        )

        # Normals per vertex (approximate, not per-face)
        normals_coords = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=np.float32,
        )
        normals_coords = normals_coords / np.linalg.norm(normals_coords, axis=1, keepdims=True)

        geometry = Geometry(vertices, faces_indices, uvs_coords, normals_coords)
        return geometry

    @staticmethod
    def grid(width: float = 1.0, depth: float = 1.0, rows: int = 10, columns: int = 10) -> Geometry:
        vertices = []
        faces_indices = []
        uvs_coords = []
        normals_coords = []

        # Create vertices, uvs and normals
        for row_index in range(rows + 1):
            for col_index in range(columns + 1):
                x = (col_index / columns - 0.5) * width
                z = (row_index / rows - 0.5) * depth
                vertices.append([x, 0.0, z])
                uvs_coords.append([col_index / columns, row_index / rows])
                normals_coords.append([0.0, 1.0, 0.0])  # Upward normal

        vertices = np.array(vertices).astype(np.float32)
        uvs_coords = np.array(uvs_coords).astype(np.float32)
        normals_coords = np.array(normals_coords).astype(np.float32)

        # Create faces (two triangles per grid cell)
        for row_index in range(rows):
            for col_index in range(columns):
                top_left = row_index * (columns + 1) + col_index
                top_right = top_left + 1
                bottom_left = (row_index + 1) * (columns + 1) + col_index
                bottom_right = bottom_left + 1

                faces_indices.append([top_left, bottom_left, top_right])
                faces_indices.append([top_right, bottom_left, bottom_right])

        faces_indices = np.array(faces_indices).astype(np.float32)

        geometry = Geometry(vertices, faces_indices, uvs_coords, normals_coords)
        return geometry
