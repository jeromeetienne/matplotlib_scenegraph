from matplotlib import lines
from pyrr import vector3
import numpy as np
from ..core.object_3d import Object3D
from ..core.constants import Constants
from ..core.geometry import Geometry


class Lines(Object3D):
    __slots__ = ("geometry", "color")

    def __init__(self, geometry: Geometry, color: np.ndarray = Constants.Color.GRAY) -> None:
        super().__init__()

        # sanity checks
        assert len(geometry.vertices) % 2 == 0, f"Lines vertices length must be even, got {len(geometry.vertices)}"

        self.name = f"a {Lines.__name__}"
        self.geometry: Geometry = geometry
        self.color: np.ndarray = color

    @staticmethod
    def from_mesh_geometry(mesh_geometry: Geometry) -> "Lines":
        """
        Create a Lines object from a mesh Geometry (with faces).
        Each edge of each face will become a line segment.
        """
        # sanity check
        assert mesh_geometry.indices is not None, "The mesh geometry MUST contain face indices"

        # Get info from the geometry
        face_count = mesh_geometry.indices.shape[0]
        vertices_per_face = mesh_geometry.indices.shape[1]
        vertices_per_line = 2

        # Each face has vertices_per_face edges, each edge has 2 vertices
        line_vertices = np.zeros((face_count * vertices_per_face * vertices_per_line, 3)).astype(np.float32)

        # Create line vertices from the mesh faces
        for face_index in range(face_count):
            for vertex_index in range(vertices_per_face):

                indice_start = mesh_geometry.indices[face_index, vertex_index]
                indice_end = mesh_geometry.indices[face_index, (vertex_index + 1) % vertices_per_face]

                vertex_start = mesh_geometry.vertices[int(indice_start)]
                vertex_end = mesh_geometry.vertices[int(indice_end)]

                line_vertices[(face_index * vertices_per_face + vertex_index) * 2] = vertex_start
                line_vertices[(face_index * vertices_per_face + vertex_index) * 2 + 1] = vertex_end

        # Build the lines object
        lines_geometry = Geometry(line_vertices)
        lines = Lines(lines_geometry)

        return lines
