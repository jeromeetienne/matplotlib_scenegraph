# pip imports
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core.constants import Constants
from ..core.geometry import Geometry


class Points(Object3D):
    __slots__ = (
        "geometry",
        "colors",
        "sizes",
        "edge_colors",
        "edge_widths",
    )

    def __init__(
        self,
        geometry: Geometry,
        color: np.ndarray = Constants.Color.GRAY,
        sizes: np.ndarray = np.array([10.0]),
        edge_colors: np.ndarray = Constants.Color.BLACK,
        edge_widths: np.ndarray = np.array([1.0]),
    ) -> None:
        super().__init__()

        self.name = f"a {Points.__name__}"
        self.geometry: Geometry = geometry
        self.colors: np.ndarray = color
        self.sizes: np.ndarray = sizes
        self.edge_colors: np.ndarray = edge_colors
        self.edge_widths: np.ndarray = edge_widths
