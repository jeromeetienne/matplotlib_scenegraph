# pip imports
import numpy as np

# local imports
from mpl_graph.geometry import Geometry, GeometryUtils, MeshGeometry


class GeometryShape:
    @staticmethod
    def plane(width: float = 1.0, height: float = 1.0) -> MeshGeometry:
        half_width = width / 2.0
        half_height = height / 2.0
        vertices = np.array(
            [
                [-half_width, -half_height, 0.0],
                [half_width, -half_height, 0.0],
                [half_width, half_height, 0.0],
                [-half_width, half_height, 0.0],
            ]
        )
        faces_indices = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        uvs_coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        normals_coords = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        mesh_geometry = MeshGeometry(vertices, faces_indices, uvs_coords, normals_coords)
        mesh_geometry = GeometryUtils.expand_vertices(mesh_geometry)
        return mesh_geometry

    @staticmethod
    def box(
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        width_segments: int = 1,
        height_segments: int = 1,
        depth_segments: int = 1,
    ) -> MeshGeometry:
        width_segments = max(1, int(width_segments))
        height_segments = max(1, int(height_segments))
        depth_segments = max(1, int(depth_segments))

        half_w = width / 2.0
        half_h = height / 2.0
        half_d = depth / 2.0

        vertices = []
        faces_indices = []
        uvs_coords = []
        normals_coords = []
        vertex_offset = 0

        def build_face(origin: np.ndarray, u_vec: np.ndarray, v_vec: np.ndarray, normal: np.ndarray, seg_u: int, seg_v: int) -> None:
            nonlocal vertex_offset

            u_step = u_vec / seg_u
            v_step = v_vec / seg_v

            for v_index in range(seg_v + 1):
                for u_index in range(seg_u + 1):
                    position = origin + u_step * u_index + v_step * v_index
                    vertices.append(position)
                    uvs_coords.append([u_index / seg_u, v_index / seg_v])
                    normals_coords.append(normal)

            stride = seg_u + 1
            for v_index in range(seg_v):
                for u_index in range(seg_u):
                    a = vertex_offset + u_index + stride * v_index
                    b = vertex_offset + u_index + stride * (v_index + 1)
                    c = vertex_offset + (u_index + 1) + stride * (v_index + 1)
                    d = vertex_offset + (u_index + 1) + stride * v_index

                    faces_indices.append([a, d, b])
                    faces_indices.append([d, c, b])

            vertex_offset += (seg_u + 1) * (seg_v + 1)

        # front (+Z)
        build_face(
            origin=np.array([-half_w, -half_h, half_d], dtype=np.float32),
            u_vec=np.array([width, 0.0, 0.0], dtype=np.float32),
            v_vec=np.array([0.0, height, 0.0], dtype=np.float32),
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            seg_u=width_segments,
            seg_v=height_segments,
        )

        # back (-Z)
        build_face(
            origin=np.array([half_w, -half_h, -half_d], dtype=np.float32),
            u_vec=np.array([-width, 0.0, 0.0], dtype=np.float32),
            v_vec=np.array([0.0, height, 0.0], dtype=np.float32),
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            seg_u=width_segments,
            seg_v=height_segments,
        )

        # top (+Y)
        build_face(
            origin=np.array([-half_w, half_h, half_d], dtype=np.float32),
            u_vec=np.array([width, 0.0, 0.0], dtype=np.float32),
            v_vec=np.array([0.0, 0.0, -depth], dtype=np.float32),
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            seg_u=width_segments,
            seg_v=depth_segments,
        )

        # bottom (-Y)
        build_face(
            origin=np.array([-half_w, -half_h, -half_d], dtype=np.float32),
            u_vec=np.array([width, 0.0, 0.0], dtype=np.float32),
            v_vec=np.array([0.0, 0.0, depth], dtype=np.float32),
            normal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            seg_u=width_segments,
            seg_v=depth_segments,
        )

        # right (+X)
        build_face(
            origin=np.array([half_w, -half_h, half_d], dtype=np.float32),
            u_vec=np.array([0.0, 0.0, -depth], dtype=np.float32),
            v_vec=np.array([0.0, height, 0.0], dtype=np.float32),
            normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            seg_u=depth_segments,
            seg_v=height_segments,
        )

        # left (-X)
        build_face(
            origin=np.array([-half_w, -half_h, -half_d], dtype=np.float32),
            u_vec=np.array([0.0, 0.0, depth], dtype=np.float32),
            v_vec=np.array([0.0, height, 0.0], dtype=np.float32),
            normal=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            seg_u=depth_segments,
            seg_v=height_segments,
        )

        vertices_array = np.asarray(vertices, dtype=np.float32)
        indices_array = np.asarray(faces_indices, dtype=np.uint32)
        uvs_array = np.asarray(uvs_coords, dtype=np.float32)
        normals_array = np.asarray(normals_coords, dtype=np.float32)

        mesh_geometry = MeshGeometry(vertices_array, indices_array, uvs_array, normals_array)
        mesh_geometry = GeometryUtils.expand_vertices(mesh_geometry)
        return mesh_geometry

    @staticmethod
    def grid(width: float = 1.0, depth: float = 1.0, rows: int = 10, columns: int = 10) -> MeshGeometry:
        vertices = []
        faces_indices = []
        uvs_coords = []
        normals_coords = []

        # Create vertices, uvs and normals
        for row_index in range(rows + 1):
            for col_index in range(columns + 1):
                x = (col_index / columns - 0.5) * width
                z = (row_index / rows - 0.5) * depth
                vertices.append([x, 0.0, z])
                uvs_coords.append([col_index / columns, row_index / rows])
                normals_coords.append([0.0, 1.0, 0.0])  # Upward normal

        vertices = np.array(vertices).astype(np.float32)
        uvs_coords = np.array(uvs_coords).astype(np.float32)
        normals_coords = np.array(normals_coords).astype(np.float32)

        # Create faces (two triangles per grid cell)
        for row_index in range(rows):
            for col_index in range(columns):
                top_left = row_index * (columns + 1) + col_index
                top_right = top_left + 1
                bottom_left = (row_index + 1) * (columns + 1) + col_index
                bottom_right = bottom_left + 1

                faces_indices.append([top_left, bottom_left, top_right])
                faces_indices.append([top_right, bottom_left, bottom_right])

        faces_indices = np.array(faces_indices).astype(np.uint32)

        mesh_geometry = MeshGeometry(vertices, faces_indices, uvs_coords, normals_coords)
        # Expand the vertices
        mesh_geometry = GeometryUtils.expand_vertices(mesh_geometry)
        return mesh_geometry
