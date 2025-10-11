# pip imports
from typing_extensions import deprecated
import numpy as np
import matplotlib.pyplot

# local imports
from core.object_3d import Object3D


class TexturedMesh(Object3D):

    def __init__(self, faces_indices: np.ndarray, vertices_coords: np.ndarray, uvs_coords: np.ndarray, texture: np.ndarray):
        super().__init__()

        # sanity check - make sure we have triangular faces
        assert faces_indices.ndim == 2 and faces_indices.shape[1] == 3, "faces_indices should be of shape [M, 3]"
        assert vertices_coords.ndim == 2 and vertices_coords.shape[1] == 3, "vertices_coords should be of shape [N, 3]"
        assert uvs_coords.ndim == 2 and uvs_coords.shape[1] == 2, "uvs_coords should be of shape [N, 2]"
        assert len(vertices_coords) == len(uvs_coords), "The number of vertices must be equal to the number of texture coordinates"
        assert texture.ndim == 3 and texture.shape[2] in [3, 4], "texture should be of shape [H, W, 3] or [H, W, 4]"

        # copy the arguments
        # TODO to change in .triangle_count  and reshape vertices to (M, 3, 3)
        # - store .vertices and .uvs as (N, 3) and (N, 2)
        self.faces_vertices = vertices_coords[faces_indices]
        self.faces_uvs = uvs_coords[faces_indices]
        self.texture = texture

        # remove the alpha channel if any
        texture = texture[::-1, ::1, :3]

        assert self.faces_vertices.ndim == 3 and self.faces_vertices.shape[1:] == (3, 3), "faces_vertices should be of shape [M, 3, 3]"
        assert self.faces_uvs.ndim == 3 and self.faces_uvs.shape[1:] == (3, 2), "faces_uvs should be of shape [M, 3, 2]"
