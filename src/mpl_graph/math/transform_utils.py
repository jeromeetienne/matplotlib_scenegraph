# pip imports
import numpy as np
from pyrr import matrix44

# local imports
from ..cameras.camera_base import CameraBase
from ..core.object_3d import Object3D


class TransformUtils:
    @staticmethod
    def compute_mvp_matrix(camera: CameraBase, object3d: Object3D) -> np.ndarray:
        """
        Compute the Model-View-Projection (MVP) matrix for a 3D object.

        # Useful resources:
        - https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix.html
        - http://www.codinglabs.net/article_world_view_projection_matrix.aspx
        - https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_model_view_projection
        """
        # Build the model, view, projection matrices
        model_matrix = object3d.get_world_matrix()
        if False:
            # Custom camera matrix (look at) - put it in Object3D
            camera_position = camera.get_world_position()
            camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            camera_matrix = matrix44.create_look_at(eye=camera_position, target=camera_target, up=camera_up)
            view_matrix = matrix44.inverse(camera_matrix)
        else:
            view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()

        # compute full transform, first model, then view, then projection
        mvp_matrix = matrix44.create_identity(dtype=np.float32)
        mvp_matrix = matrix44.multiply(mvp_matrix, model_matrix)
        mvp_matrix = matrix44.multiply(mvp_matrix, view_matrix)
        mvp_matrix = matrix44.multiply(mvp_matrix, projection_matrix)

        return mvp_matrix

    # @staticmethod
    # def apply_transform(vertices: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    #     # sanity checks
    #     assert vertices.shape[1] == 3 and vertices.ndim == 2, f"vertices should be of shape [N, 3]. Got {vertices.shape}"
    #     assert transform_matrix.shape == (4, 4), f"transform should be of shape [4, 4]. Got {transform_matrix.shape}"

    #     # make vertices homogeneous
    #     vertices_hom = np.hstack([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)])  # [N, 4]

    #     # apply full transform to homogeneous vertices
    #     vertices_world_hom = vertices_hom @ transform_matrix  # [N, 4]

    #     # get w for perspective divide
    #     vertices_w = vertices_world_hom[:, 3:4]  # [N, 1]

    #     # drop w for clip space
    #     vertices_clip = vertices_world_hom[:, :3]  # [N, 3]

    #     # avoid division by zero
    #     vertices_w[vertices_w == 0] = 1e-6

    #     # Perform perspective divide to get normalized device coordinates (NDC)
    #     vertices_ndc = vertices_clip / vertices_w  # [N, 3]

    #     return vertices_ndc

    # @staticmethod
    # def normalize_vertices_to_unit_cube(vertices: np.ndarray) -> np.ndarray:
    #     # sanity checks
    #     assert vertices.shape[1] == 3 and vertices.ndim == 2, "vertices should be of shape [N, 3]"
    #     # make a copy to avoid modifying the original
    #     vertices_normalized = vertices.copy()
    #     # find the min and max of each axis
    #     min_coords = vertices_normalized.min(axis=0)
    #     max_coords = vertices_normalized.max(axis=0)
    #     vertex_center = (min_coords + max_coords) / 2.0
    #     # center the vertices
    #     vertices_normalized -= vertex_center
    #     # findout the max extent
    #     scale = np.abs(vertices_normalized).max()
    #     # scale to fit in unit cube
    #     vertices_normalized /= scale
    #     # now the vertices should be in the range [-1, 1]
    #     return vertices_normalized
