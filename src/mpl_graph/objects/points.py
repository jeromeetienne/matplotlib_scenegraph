from ..core.object_3d import Object3D
from pyrr import vector3
import numpy as np
from ..core.constants import Constants


class Points(Object3D):
    __slots__ = (
        "vertices",
        "colors",
        "sizes",
        "edge_colors",
        "edge_widths",
    )

    def __init__(
        self,
        vertices: np.ndarray = vector3.create(),
        color: np.ndarray = Constants.Color.GRAY,
        sizes: np.ndarray = np.array([10.0]),
        edge_colors: np.ndarray = Constants.Color.BLACK,
        edge_widths: np.ndarray = np.array([1.0]),
    ) -> None:
        super().__init__()

        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must be a Nx3 array"

        self.name = f"a {Points.__name__}"
        self.vertices: np.ndarray = vertices
        self.colors: np.ndarray = color
        self.sizes: np.ndarray = sizes
        self.edge_colors: np.ndarray = edge_colors
        self.edge_widths: np.ndarray = edge_widths
