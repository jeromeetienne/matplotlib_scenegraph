# pip imports
import numpy as np
import meshio

# local imports
from mpl_graph.geometry import MeshGeometry
from mpl_graph.objects.polygons import Polygons
from mpl_graph.core.transform_utils import TransformUtils


class MeshUtils:
    """
    Manual Wavefront .obj file parser
    """

    # =============================================================================
    #
    # =============================================================================

    @staticmethod
    def parse_obj_file_manual(file_path: str) -> MeshGeometry:
        """
        Parse a Wavefront .obj file and extract vertex, texture, and normal coordinates.

        Arguments:
            file_path (str): Path to the .obj file.

        Returns:
            tuple: A tuple containing:
                - vertices_coords (np.ndarray): Array of vertex coordinates. Shape (N, 3).
                - faces_indices (np.ndarray): Array of face indices. Shape (M, 3).
                - uvs_coords (np.ndarray | None): Array of texture coordinates (if available). Shape (N, 2) or None.
                - normals_coords (np.ndarray | None): Array of normal coordinates (if available). Shape (N, 3) or None.
        """

        # =============================================================================
        # Parse the .obj
        # =============================================================================

        # Parse the .obj file using the plain parser
        vertices_coords, uvs_coords, normals_coords, faces_vertex_indices, faces_uv_indices, faces_normal_indices = MeshUtils.parse_obj_plain(file_path)

        # Sanity checks - a valid .obj file should have at least vertices and faces
        assert len(vertices_coords) > 0, "No vertices found in the .obj file."
        assert len(faces_vertex_indices) > 0, "No faces found in the .obj file."
        assert (
            faces_vertex_indices.shape[1] == 3
        ), f"Only triangular faces are supported in this parser. Got faces with {faces_vertex_indices.shape[1]} vertices."

        # =============================================================================
        # Process the data to ensure len(vertices) == len(uvs) == len(normals)
        # =============================================================================

        vertices = vertices_coords[faces_vertex_indices].reshape(-1, 3)
        uvs = uvs_coords[faces_uv_indices].reshape(-1, 2) if faces_uv_indices is not None and uvs_coords is not None else None
        normals = normals_coords[faces_normal_indices].reshape(-1, 3) if faces_normal_indices is not None and normals_coords is not None else None
        # Recompute the faces indices
        indices = np.arange(len(vertices)).reshape(-1, 3)

        # =============================================================================
        # Check all is ok, and return the values
        # =============================================================================

        # sanity checks
        assert len(indices) > 0, "No indices found after processing the .obj file."
        assert len(vertices) > 0, "No vertices found after processing the .obj file."
        assert uvs is None or len(vertices) == len(uvs), f"vertex count is different than uv count. got vertex count {len(vertices)} and uv count {len(uvs)}"
        assert normals is None or len(vertices) == len(
            normals
        ), f"vertex coords count is different than normal coords count. got vertex count {len(vertices)} and normal count {len(normals)}"

        mesh_geometry = MeshGeometry(vertices, indices, uvs, normals)

        # return the values
        return mesh_geometry

    # =============================================================================
    #
    # =============================================================================

    @staticmethod
    def parse_obj_plain(filename: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Read a wavefront filename and returns vertices, texcoords and
        respective indices for faces and texcoords

        Arguments:
            filename (str): Path to the .obj file.

        Returns:
            tuple: A tuple containing:
                - vertices_coords (np.ndarray): Array of vertex coordinates. Shape (N, 3).
                - uvs_coords (np.ndarray): Array of texture coordinates. Shape (N, 2).
                - normals_coords (np.ndarray): Array of normal coordinates. Shape (N, 3).
                - faces_vertex_indices (np.ndarray): Array of face vertex indices. Shape (M, 3).
                - faces_uv_indices (np.ndarray): Array of face texture coordinate indices. Shape (M, 3).
                - faces_normal_indices (np.ndarray): Array of face normal coordinate indices. Shape (M, 3).
        """

        vertices_coords, uvs_coords, normals_coords, faces_vertex_indices, faces_uv_indices, faces_normal_indices = [], [], [], [], [], []
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip()
                # skip empty lines
                if len(line) == 0:
                    continue
                # skip comments
                if line.startswith("#"):
                    continue
                # split the line into values
                values = line.split()
                # parse the values
                if values[0] == "v":
                    vertices_coords.append([float(x) for x in values[1:4]])
                elif values[0] == "vt":
                    uvs_coords.append([float(x) for x in values[1:3]])
                elif values[0] == "vn":
                    normals_coords.append([float(x) for x in values[1:4]])
                elif values[0] == "f":
                    face_vertices = values[1:]
                    assert len(face_vertices) == 3, "Only triangular faces are supported in this parser"
                    face_splitted_indices = [indices.split("/") for indices in face_vertices]
                    if len(face_splitted_indices[0]) >= 1:
                        # vertex indices
                        faces_vertex_indices.append([face_splitted_indices[index][0] for index in range(3)])
                    if len(face_splitted_indices[1]) >= 2:
                        # vertex indice / uv indice
                        faces_uv_indices.append([face_splitted_indices[index][1] for index in range(3)])
                    if len(face_splitted_indices[2]) >= 3:
                        # vertex indice / uv indice / normal indice
                        faces_normal_indices.append([face_splitted_indices[index][2] for index in range(3)])

        # sanity checks
        assert len(vertices_coords) > 0, "No vertices found in the .obj file"

        # convert to numpy arrays or None
        vertices_coords = np.array(vertices_coords, dtype=np.float32)
        uvs_coords = np.array(uvs_coords, dtype=np.float32) if len(uvs_coords) > 0 else None
        normals_coords = np.array(normals_coords, dtype=np.float32) if len(normals_coords) > 0 else None
        faces_vertex_indices = np.array(faces_vertex_indices, dtype=np.int32) - 1
        faces_uv_indices = (np.array(faces_uv_indices, dtype=np.int32) - 1) if len(faces_uv_indices) > 0 else None
        faces_normal_indices = (np.array(faces_normal_indices, dtype=np.int32) - 1) if len(faces_normal_indices) > 0 else None

        assert np.max(faces_vertex_indices) <= len(vertices_coords), "Face vertex index out of range"
        if faces_uv_indices is not None and uvs_coords is not None:
            assert np.max(faces_uv_indices) <= len(uvs_coords), "Face uv index out of range"
        if faces_normal_indices is not None and normals_coords is not None:
            assert np.max(faces_normal_indices) <= len(normals_coords), "Face normal index out of range"

        # return the values
        return vertices_coords, uvs_coords, normals_coords, faces_vertex_indices, faces_uv_indices, faces_normal_indices

    # =============================================================================
    #
    # =============================================================================

    @staticmethod
    def parse_obj_file_meshio(
        file_path: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Parse a Wavefront .obj file and extract vertex, texture, and normal coordinates.

        Arguments:
            file_path (str): Path to the .obj file.

        Returns:
            tuple: A tuple containing:
                - vertices_coords (np.ndarray): Array of vertex coordinates. Shape (N, 3).
                - faces_indices (np.ndarray): Array of face indices. Shape (M, 3).
                - faces_uvs (np.ndarray | None): Array of texture coordinates (if available). Shape (N, 2) or None.
                - faces_normals (np.ndarray | None): Array of normal coordinates (if available). Shape (N, 3) or None.
        """

        meshio_mesh = meshio.read(file_path)
        vertices_coords = meshio_mesh.points
        # only 3d triangular meshes are supported for now
        faces_indices = meshio_mesh.cells[0].data

        faces_vertices = vertices_coords[faces_indices]
        # Optional texture coordinates
        faces_uvs = meshio_mesh.point_data["obj:vt"] if "obj:vt" in meshio_mesh.point_data else None
        # Optional (per-vertex) normals coordinates
        faces_normals = meshio_mesh.point_data["obj:vn"] if "obj:vn" in meshio_mesh.point_data else None

        return vertices_coords, faces_indices, faces_uvs, faces_normals
