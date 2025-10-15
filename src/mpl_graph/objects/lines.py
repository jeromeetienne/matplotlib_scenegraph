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
        edges_set = set()
        lines_vertices: list[np.ndarray] = []

        # Create line vertices from the mesh faces
        for face_index in range(face_count):
            for vertex_index in range(vertices_per_face):

                index_start = mesh_geometry.indices[face_index, vertex_index]
                index_end = mesh_geometry.indices[face_index, (vertex_index + 1) % vertices_per_face]

                # get the vertex positions
                vertex_start = np.asarray(mesh_geometry.vertices[int(index_start)], dtype=np.float32)
                vertex_end = np.asarray(mesh_geometry.vertices[int(index_end)], dtype=np.float32)

                # build a canonical key based on coordinates so orientation and indices do not matter
                if dedup_edges:
                    start_tuple = tuple(float(value) for value in vertex_start)
                    end_tuple = tuple(float(value) for value in vertex_end)
                    edge_key = (start_tuple, end_tuple) if start_tuple <= end_tuple else (end_tuple, start_tuple)
                    if edge_key in edges_set:
                        continue
                    edges_set.add(edge_key)

                lines_vertices.append(vertex_start)
                lines_vertices.append(vertex_end)

        lines_vertices_np = np.asanyarray(lines_vertices, dtype=np.float32)
        # Build the lines object
        lines_geometry = Geometry(lines_vertices_np)
        lines = Lines(lines_geometry)

        return lines
