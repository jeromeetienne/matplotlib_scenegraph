# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import numpy as np
from pyrr import matrix44

# local imports
from objects.points import Points
from renderers.matplotlib.renderer import RendererMatplotlib
from cameras.camera_orthographic import CameraOrthographic
from cameras.camera_base import CameraBase
from core.object_3d import Object3D


class TransformUtils:
    @staticmethod
    def compute_full_transform(camera: CameraBase, object3d: Object3D) -> np.ndarray:
        camera_position = camera.get_world_position()
        camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        camera_matrix = matrix44.create_look_at(eye=camera_position, target=camera_target, up=camera_up)

        # View matrix is inverse of camera world matrix
        view_matrix = matrix44.inverse(camera_matrix)
        model_matrix = object3d.get_world_matrix()
        # view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()

        temp = matrix44.multiply(view_matrix, model_matrix)
        full_transform = matrix44.multiply(projection_matrix, temp)
        return full_transform

    @staticmethod
    def apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
        # sanity checks
        assert vertices.shape[1] == 3 and vertices.ndim == 2, f"vertices should be of shape [N, 3]. Got {vertices.shape}"
        assert transform.shape == (4, 4), f"transform should be of shape [4, 4]. Got {transform.shape}"

        # make vertices homogeneous
        vertices_hom = np.hstack([vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)])  # [N, 4]
        # apply full transform to vertices
        vertices_world_hom = vertices_hom @ transform  # [N, 4]
        # vertices_world_hom = vertices_hom @ transform  # [N, 4]
        # drop w for clip space
        vertices_clip = vertices_world_hom[:, :3]  # [N, 3]
        # Perform perspective divide to get normalized device coordinates (NDC)
        vertices_transformed = vertices_clip / vertices_world_hom[:, 3:4]  # [N, 3]

        return vertices_transformed

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
