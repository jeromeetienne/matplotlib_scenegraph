from matplotlib import lines
from core.object_3d import Object3D
from pyrr import vector3
import numpy as np
from core.constants import Constants


class Lines(Object3D):
    def __init__(
        self,
        vertices: np.ndarray = vector3.create(),
        color: np.ndarray = Constants.GRAY,
    ) -> None:
        super().__init__()

        line_count = len(vertices) // 2

        # sanity checks
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must be a Nx3 array"
        assert line_count * 2 == len(vertices), "Lines vertices length must be even"

        self.vertices: np.ndarray = vertices
        self.color: np.ndarray = color
