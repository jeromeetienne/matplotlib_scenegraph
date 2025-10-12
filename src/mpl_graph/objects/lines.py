from matplotlib import lines
from pyrr import vector3
import numpy as np
from ..core.object_3d import Object3D
from ..core.constants import Constants
from ..core.geometry import Geometry


class Lines(Object3D):
    def __init__(self, geometry: Geometry, color: np.ndarray = Constants.Color.GRAY) -> None:
        super().__init__()

        # sanity checks
        assert len(geometry.vertices) % 2 == 0, f"Lines vertices length must be even, got {len(geometry.vertices)}"

        self.name = f"a {Lines.__name__}"
        self.geometry: Geometry = geometry
        self.color: np.ndarray = color
