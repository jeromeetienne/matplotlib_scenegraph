# pip imports
import numpy as np
from pyrr import vector3, vector4

# local imports
from .geometry import Geometry


class MeshGeometry(Geometry):
    __slot__ = ("indices", "uvs", "normals")

    def __init__(
        self,
        vertices: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        uvs: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ):
        """
        A class representing a 3D geometry with vertices, faces, texture coordinates, and normals.
        Only triangular faces are supported.

        Arguments:
            vertices (np.ndarray): array of vertex coordinates, shape (N, 3)
            indices (np.ndarray | None): array of face indices, shape (M, 3) or None if there is no face
            uvs (np.ndarray | None): array of texture coordinates, shape (N, 2) or None if there is no texture
            normals (np.ndarray | None): array of normal coordinates, shape (N, 3) or None if there is no normal
        """

        # assign attributes
        self.vertices: np.ndarray = vertices if vertices is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape((1, 3))
        """array of vertex coordinates, shape (N, 3)"""

        self.indices: np.ndarray = indices if indices is not None else np.array([0, 0, 0], dtype=np.int32).reshape((1, 3))
        """array of face indices, shape (M, 3) or None if there is no face"""

        self.uvs: np.ndarray | None = uvs
        """array of texture coordinates, shape (N, 2) or None if there is no texture"""

        self.normals: np.ndarray | None = normals
        """array of normal coordinates, shape (N, 3) or None if there is no normal"""

        # sanity check - make sure we have triangular faces
        assert len(self.indices) > 0, f"The geometry must have at least one face, got {len(self.indices)}"
        assert self.indices.ndim == 2 and self.indices.shape[1] == 3, f"indices should be of shape [M, 3], got {self.indices.shape}"
        if self.uvs is not None:
            assert self.uvs.ndim == 2 and self.uvs.shape[1] == 2, f"uvs_coords should be of shape [N, 2], got {self.uvs.shape}"
            assert len(self.uvs) == len(self.vertices), "The number of uvs must be equal to the number of vertices"
        if self.normals is not None:
            assert self.normals.ndim == 2 and self.normals.shape[1] == 3, f"normals should be of shape [N, 3], got {self.normals.shape}"
            assert len(self.normals) == len(self.vertices), "The number of normals must be equal to the number of vertices"
