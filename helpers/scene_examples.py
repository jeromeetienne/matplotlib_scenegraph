# stdlib imports
import os

# pip imports
import numpy as np
import matplotlib.image

# local imports
from helpers.mesh_parser_obj_manual import MeshParserObjManual
from objects.points import Points
from core.object_3d import Object3D
from core.constants import Constants
from core.transform_utils import TransformUtils
from objects.textured_mesh import TexturedMesh
from helpers.animation_loop import AnimationLoop
from objects.polygons import Polygons


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(data_path, "models")
images_path = os.path.join(data_path, "images")


class SceneExamples:

    @staticmethod
    def polygons_from_obj(file_path: str) -> Polygons:
        faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(file_path)

        vertices_coords = TransformUtils.normalize_vertices_to_unit_cube(vertices_coords)

        vertices = vertices_coords[faces_indices]
        polygon_count = vertices.shape[0]
        vertices_per_polygon = vertices.shape[1]
        vertices = vertices.reshape(polygon_count * vertices_per_polygon, 3)
        polygons = Polygons(vertices, polygon_count, vertices_per_polygon)

        # polygons = Points(vertices)
        return polygons

    @staticmethod
    def addAnimatedModels(model_root: Object3D, animation_loop: AnimationLoop) -> None:
        cube_points = SceneExamples.getCubePoints()
        model_root.add_child(cube_points)
        cube_points.scale[:] = 0.1
        cube_points.position[0] = -3
        cube_points.position[1] = 3

        bunny_points = SceneExamples.getBunnyPoints()
        model_root.add_child(bunny_points)
        bunny_points.scale[:] = 0.2
        bunny_points.position[0] = 3
        bunny_points.position[1] = 3

        head_points = SceneExamples.getHeadPoints()
        model_root.add_child(head_points)
        head_points.scale[:] = 0.1
        head_points.position[0] = 0
        head_points.position[1] = -3

        def update_model_root(delta_time: float, timestamp: float) -> list[Object3D]:
            range = np.sin(timestamp) * 1 + 2
            bunny_points.position[1] = np.abs(np.cos(timestamp * 5) * range)
            return [bunny_points]

        animation_loop.add_callback(update_model_root)

    @staticmethod
    def addRandomPoints(point_count: int = 1000) -> Points:
        points = np.random.uniform(-1, 1, (point_count, 3))
        colors = np.random.uniform(0, 1, (point_count, 4))
        colors[:, 3] = 1.0  # set alpha to 1
        random_points = Points(points, color=colors)
        return random_points

    @staticmethod
    def getBunnyPoints() -> Points:
        face_indices, vertex_coords, uv_coords, normal_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "bunny.obj"))
        vertex_coords = TransformUtils.normalize_vertices_to_unit_cube(vertex_coords)
        points_bunny = Points(vertex_coords, color=Constants.PURPLE)
        return points_bunny

    @staticmethod
    def getCubePoints() -> Points:
        face_indices, vertex_coords, uv_coords, normal_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "cube_meshio.obj"))
        vertex_coords = TransformUtils.normalize_vertices_to_unit_cube(vertex_coords)
        points_bunny = Points(vertex_coords, color=Constants.CYAN)
        return points_bunny

    @staticmethod
    def getHeadPoints() -> Points:
        faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "head_meshio.obj"))
        vertices_coords = TransformUtils.normalize_vertices_to_unit_cube(vertices_coords)
        points_bunny = Points(vertices_coords, color=Constants.CYAN)
        return points_bunny

    @staticmethod
    def getHeadTexturedMesh() -> TexturedMesh:
        texture = SceneExamples.getUvGridTexture()

        obj_path = os.path.join(models_path, "head_meshio.obj")
        faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(obj_path)
        assert uvs_coords is not None, "The .obj file must contain texture coordinates (vt)"
        textured_mesh = TexturedMesh(faces_indices, vertices_coords, uvs_coords, texture)
        textured_mesh.name = "TexturedMesh"
        return textured_mesh

    @staticmethod
    def getUvGridTexture() -> np.ndarray:
        texture_path = os.path.join(images_path, "uv-grid.png")
        texture = SceneExamples._load_texture(texture_path)
        return texture

    @staticmethod
    def _load_texture(file_path: str) -> np.ndarray:
        """
        Load a texture image from file.

        Arguments:
            file_path (str): Path to the image file.
        Returns:
            np.ndarray: Loaded image as a numpy array.
        """
        texture = matplotlib.image.imread(file_path)
        if texture.dtype != np.uint8:
            # convert to uint8
            texture = (texture * 255).astype(np.uint8)

        return texture
