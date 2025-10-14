from matplotlib import lines
from pyrr import vector3
import numpy as np
from ..core.object_3d import Object3D
from ..core.constants import Constants
from ..geometry import Geometry, MeshGeometry
from ..materials.lines_material import LinesMaterial


class Lines(Object3D):
    __slots__ = ("geometry", "material")

    def __init__(self, geometry: Geometry | None = None, material: LinesMaterial | None = None) -> None:
        """
        Create a Lines object.
        - each line segment has 2 vertices
        - the number of vertices in the geometry must be even
        """
        super().__init__()

        self.name = f"a {Lines.__name__}"
        self.geometry: Geometry = geometry if geometry is not None else Geometry()
        self.material: LinesMaterial = material if material is not None else LinesMaterial()

        # sanity checks
        assert len(self.geometry.vertices) % 2 == 0, f"Lines vertices length must be even, got {len(self.geometry.vertices)}"

    @staticmethod
    def from_mesh_geometry(mesh_geometry: MeshGeometry, dedup_edges: bool = True) -> "Lines":
        """
        Create a Lines object from a mesh Geometry (with faces).
        Each edge of each face will become a line segment.

        Arguments:
            - mesh_geometry (MeshGeometry): the input mesh geometry (with faces)
            - dedup_edges (bool): if True, duplicate edges will be removed (default: True)
        """
        # sanity check
        assert mesh_geometry.indices is not None, "The mesh geometry MUST contain face indices"

        # Get info from the geometry
        face_count = mesh_geometry.indices.shape[0]
        vertices_per_face = mesh_geometry.indices.shape[1]
        vertices_per_line = 2

        # Each face has vertices_per_face edges, each edge has 2 vertices
        line_vertices = np.zeros((face_count * vertices_per_face * vertices_per_line, 3)).astype(np.float32)

        # create a set which will contain the unique edges (index_start, index_end)
        edges_set = set()

        # Create line vertices from the mesh faces
        for face_index in range(face_count):
            for vertex_index in range(vertices_per_face):

                index_start = mesh_geometry.indices[face_index, vertex_index]
                index_end = mesh_geometry.indices[face_index, (vertex_index + 1) % vertices_per_face]

                # to avoid duplicating edges, we store them in a set with sorted indices
                if dedup_edges:
                    edge = (min(index_start, index_end), max(index_start, index_end))
                    if edge in edges_set:
                        continue
                    edges_set.add(edge)

                # get the vertex positions
                vertex_start = mesh_geometry.vertices[int(index_start)]
                vertex_end = mesh_geometry.vertices[int(index_end)]

                # set the line vertices
                line_vertices[(face_index * vertices_per_face + vertex_index) * 2] = vertex_start
                line_vertices[(face_index * vertices_per_face + vertex_index) * 2 + 1] = vertex_end

        # Build the lines object
        lines_geometry = Geometry(line_vertices)
        lines = Lines(lines_geometry)

        return lines
