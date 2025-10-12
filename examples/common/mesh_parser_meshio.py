# pip imports
import meshio
import numpy as np
import matplotlib.image


class MeshParserMeshio:
    """
    Mesh parser using the meshio library
    """

    @staticmethod
    def parse_obj_file(
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
