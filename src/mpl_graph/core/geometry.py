# pip imports
import numpy as np
from pyrr import vector3, vector4


class Geometry:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray | None = None, uvs: np.ndarray | None = None, normals: np.ndarray | None = None):
        """
        A class representing a 3D geometry with vertices, faces, texture coordinates, and normals.
        Only triangular faces are supported.

        Arguments:
        - vertices: array of vertex coordinates, shape (N, 3)
        - indices: array of face indices, shape (M, 3) or None if there is no face
        - uvs: array of texture coordinates, shape (N, 2) or None if there is no texture
        - normals: array of normal coordinates, shape (N, 3) or None if there is no normal
        """

        # sanity check - make sure we have triangular faces
        assert vertices.ndim == 2 and vertices.shape[1] == 3, f"vertices_coords should be of shape [N, 3], got {vertices.shape}"
        if indices is not None:
            assert indices.ndim == 2 and indices.shape[1] == 3, f"indices should be of shape [M, 3], got {indices.shape}"
        if uvs is not None:
            assert uvs.ndim == 2 and uvs.shape[1] == 2, f"uvs_coords should be of shape [N, 2], got {uvs.shape}"
            assert len(uvs) == len(vertices), "The number of uvs must be equal to the number of vertices"
        if normals is not None:
            assert normals.ndim == 2 and normals.shape[1] == 3, f"normals should be of shape [N, 3], got {normals.shape}"
            assert len(normals) == len(vertices), "The number of normals must be equal to the number of vertices"

        # assign attributes
        self.vertices: np.ndarray = vertices
        """array of vertex coordinates, shape (N, 3)"""

        self.indices: np.ndarray | None = indices
        """array of face indices, shape (M, 3) or None if there is no face"""

        self.uvs: np.ndarray | None = uvs
        """array of texture coordinates, shape (N, 2) or None if there is no texture"""

        self.normals: np.ndarray | None = normals
        """array of normal coordinates, shape (N, 3) or None if there is no normal"""
