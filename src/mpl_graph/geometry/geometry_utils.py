# pip imports
import numpy as np

# local imports
from .geometry import Geometry


class GeometryUtils:

    # FIXME: it should handle Geometry and not only vertices
    @staticmethod
    def apply_transform(vertices: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        # sanity checks
        assert vertices.shape[1] == 3 and vertices.ndim == 2, f"vertices should be of shape [N, 3]. Got {vertices.shape}"
        assert transform_matrix.shape == (4, 4), f"transform should be of shape [4, 4]. Got {transform_matrix.shape}"

        # make vertices homogeneous
        vertices_hom = np.hstack([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)])  # [N, 4]

        # apply full transform to homogeneous vertices
        vertices_world_hom = vertices_hom @ transform_matrix  # [N, 4]

        # get w for perspective divide
        vertices_w = vertices_world_hom[:, 3:4]  # [N, 1]

        # drop w for clip space
        vertices_clip = vertices_world_hom[:, :3]  # [N, 3]

        # avoid division by zero
        vertices_w[vertices_w == 0] = 1e-6

        # Perform perspective divide to get normalized device coordinates (NDC)
        vertices_ndc = vertices_clip / vertices_w  # [N, 3]

        return vertices_ndc

    # FIXME: it should handle Geometry and not only vertices
    @staticmethod
    def normalize_vertices_to_unit_cube(vertices: np.ndarray) -> np.ndarray:
        # sanity checks
        assert vertices.shape[1] == 3 and vertices.ndim == 2, "vertices should be of shape [N, 3]"
        # make a copy to avoid modifying the original
        vertices_normalized = vertices.copy()
        # find the min and max of each axis
        min_coords = vertices_normalized.min(axis=0)
        max_coords = vertices_normalized.max(axis=0)
        vertex_center = (min_coords + max_coords) / 2.0
        # center the vertices
        vertices_normalized -= vertex_center
        # findout the max extent
        scale = np.abs(vertices_normalized).max()
        # scale to fit in unit cube
        vertices_normalized /= scale
        # now the vertices should be in the range [-1, 1]
        return vertices_normalized

    @staticmethod
    def expand_vertices(src_geometry: Geometry) -> Geometry:
        vertices = src_geometry.vertices[src_geometry.indices].reshape(-1, 3)
        uvs = src_geometry.uvs[src_geometry.indices].reshape(-1, 2) if src_geometry.uvs is not None else None
        normals = src_geometry.normals[src_geometry.indices].reshape(-1, 3) if src_geometry.normals is not None else None
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

        dst_geometry = Geometry(vertices, indices, uvs, normals)
        return dst_geometry
