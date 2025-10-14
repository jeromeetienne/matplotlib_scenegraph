# pip imports
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core.constants import Constants
from ..geometry.geometry import Geometry
from ..materials.points_material import PointsMaterial


class Points(Object3D):
    __slots__ = ("geometry", "material")

    def __init__(
        self,
        geometry: Geometry | None = None,
        material: PointsMaterial | None = None,
    ) -> None:
        super().__init__()

        self.name = f"a {Points.__name__}"
        self.geometry: Geometry = geometry if geometry is not None else Geometry()
        self.material: PointsMaterial = material if material is not None else PointsMaterial()
