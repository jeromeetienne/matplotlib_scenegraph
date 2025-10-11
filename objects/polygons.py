from matplotlib import lines
from core.object_3d import Object3D
from pyrr import vector3
import numpy as np
from core.constants import Constants


class Polygons(Object3D):
    def __init__(
        self,
        vertices: np.ndarray,
        polygon_count: int,
        vertices_per_polygon: int,
        color: np.ndarray = Constants.GRAY,
    ) -> None:
        super().__init__()

        # sanity checks
        assert vertices.ndim == 2 and vertices.shape[1] == 3, f"vertices should be of shape (N, 3). Got {vertices.shape}"
        assert polygon_count > 0, f"polygon_count should be > 0. Got {polygon_count}"
        assert vertices_per_polygon > 2, f"vertices_per_polygon should be > 2. Got {vertices_per_polygon}"
        assert len(vertices) == polygon_count * vertices_per_polygon, f"The number of vertices must be equal to polygon_count * vertices_per_polygon"

        self.polygon_count: int = polygon_count
        """number of polygons."""
        self.vertices_per_polygon: int = vertices_per_polygon
        """number of vertices per polygon."""
        self.vertices: np.ndarray = vertices.reshape(self.polygon_count * self.vertices_per_polygon, 3)
        """np.ndarray of vertices. shape (N, 3) where N is number of vertices. possible to rebuild the (P, V, 3) shape with .polygon_count and .vertices_per_polygon."""

        self.color: np.ndarray = color
