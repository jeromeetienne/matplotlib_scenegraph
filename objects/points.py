from core.object_3d import Object3D
from pyrr import vector3
import numpy as np
from core.constants import Constants


class Points(Object3D):
    def __init__(
        self,
        vertices: np.ndarray = vector3.create(),
        color: np.ndarray = Constants.GRAY,
    ) -> None:
        super().__init__()

        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must be a Nx3 array"

        self.vertices: np.ndarray = vertices
        self.color: np.ndarray = color
