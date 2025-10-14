# pip imports
from matplotlib import lines
from pyrr import vector3
import numpy as np
import typing

# local imports
from ..core import Constants, Object3D
from ..geometry import Geometry, MeshGeometry
from ..materials import PolygonsMaterial


class Polygons(Object3D):
    __slots__ = ("polygon_count", "vertices_per_polygon", "geometry", "material")

    def __init__(self, polygon_count: int, vertices_per_polygon: int, geometry: Geometry, material: PolygonsMaterial | None = None) -> None:
        """
        Create a Polygons object.
        - able to have multiple polygons with shared vertices
        - each polygon has the same number of vertices (e.g. triangles, quads, etc.)
        - each polygon can have 3 or more vertices (so triangles, quads, pentagons, etc. are all supported)
          - so it can not reuse geometry the same way as a Mesh (where all faces are triangles)
        """
        super().__init__()

        # sanity checks
        assert polygon_count > 0, f"polygon_count should be > 0. Got {polygon_count}"
        assert vertices_per_polygon > 2, f"vertices_per_polygon should be > 2. Got {vertices_per_polygon}"
        assert len(geometry.vertices) == polygon_count * vertices_per_polygon, f"The number of vertices must be equal to polygon_count * vertices_per_polygon"

        self.name = f"a {Polygons.__name__}"
        """name of the object."""

        self.polygon_count: int = polygon_count
        """number of polygons."""
        self.vertices_per_polygon: int = vertices_per_polygon
        """number of vertices per polygon."""
        self.geometry: Geometry = geometry
        """Geometry object containing the vertices."""
        self.material: PolygonsMaterial = material if material is not None else PolygonsMaterial()
        """Material object containing the material properties."""

    @staticmethod
    def from_mesh_geometry(geometry: MeshGeometry) -> "Polygons":
        """
        Create a Polygons object from a mesh MeshGeometry (with faces).
        Each face of the mesh will become a polygon.
        """
        # sanity check
        assert geometry.indices is not None, "The mesh geometry MUST contain face indices"
        # Create a polygons object
        polygon_count: int = geometry.indices.shape[0]
        vertices_per_polygon: int = geometry.indices.shape[1]
        polygons = Polygons(polygon_count, vertices_per_polygon, geometry)

        return polygons
