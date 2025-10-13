# pip imports
import numpy as np


class Geometry:
    def __init__(self, vertices: np.ndarray):
        """
        A class representing a 3D geometry with vertices.

        Arguments:
            vertices (np.ndarray): array of vertex coordinates, shape (N, 3)
        """

        # sanity check - make sure we have triangular faces
        assert vertices.ndim == 2 and vertices.shape[1] == 3, f"vertices_coords should be of shape [N, 3], got {vertices.shape}"

        # assign attributes
        self.vertices: np.ndarray = vertices
        """array of vertex coordinates, shape (N, 3)"""
