# pip imports
import numpy as np

# local imports
from objects.polygons import Polygons
from core.transform_utils import TransformUtils


class MeshParserObjManual:
    """
    Manual Wavefront .obj file parser
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
                - uvs_coords (np.ndarray | None): Array of texture coordinates (if available). Shape (N, 2) or None.
                - normals_coords (np.ndarray | None): Array of normal coordinates (if available). Shape (N, 3) or None.
        """

        vertices_coords, uvs_coords, normals_coords, faces_vertex_indices, faces_uv_indices, faces_normal_indices = MeshParserObjManual.parse_raw(file_path)

        # Sanity checks - a valid .obj file should have at least vertices and faces
        assert len(vertices_coords) > 0, "No vertices found in the .obj file."
        assert len(faces_vertex_indices) > 0, "No faces found in the .obj file."
        # TODO add more checks

        # TODO process the data to ensure len(vertices) == len(uvs) == len(normals)

        faces_indices = faces_vertex_indices
        vertices_coords = vertices_coords
        uvs_coords = uvs_coords
        normals_coords = normals_coords

        # sanity checks
        assert len(faces_indices) > 0, "No indices found after processing the .obj file."
        assert len(vertices_coords) > 0, "No vertices found after processing the .obj file."
        assert uvs_coords is None or len(vertices_coords) == len(
            uvs_coords
        ), f"vertex coords count is different than uv coords count. got vertex count {len(vertices_coords)} and uv count {len(uvs_coords)}"
        assert normals_coords is None or len(vertices_coords) == len(
            normals_coords
        ), f"vertex coords count is different than normal coords count. got vertex count {len(vertices_coords)} and normal count {len(normals_coords)}"

        # return the values
        return faces_indices, vertices_coords, uvs_coords, normals_coords

    @staticmethod
    def parse_raw(filename: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
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
