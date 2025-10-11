# pip imports
import numpy as np
from pyrr import vector3, vector4


class Geometry:
    def __init__(self):
        self.vertices: np.ndarray = np.array([], dtype=np.float32).reshape(0, 3)
        """array of vertex coordinates, shape (N, 3)"""
        self.indices: np.ndarray | None = None
        """array of face indices, shape (M, 3) or None if there is no face"""
        self.uvs: np.ndarray | None = None
        """array of texture coordinates, shape (N, 2) or None if there is no texture"""
        self.normals: np.ndarray | None = None
        """array of normal coordinates, shape (N, 3) or None if there is no normal"""
