# pip imports
import numpy as np
from pyrr import matrix44

# local imports
from ..cameras.camera import Camera
from ..core.object_3d import Object3D


class TransformUtils:
    @staticmethod
    def compute_mvp_matrix(camera: Camera, object3d: Object3D) -> np.ndarray:
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
