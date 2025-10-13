# stdlib imports
import os

# pip imports
import numpy as np

# local imports
from mpl_graph.core import Constants, Texture
from mpl_graph.geometry import Geometry, GeometryUtils
from mpl_graph.objects import Points, TexturedMesh
from .mesh_utils import MeshUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(__dirname__, "../../assets")
models_path = os.path.join(data_path, "models")
images_path = os.path.join(data_path, "images")


class SceneExamples:

    @staticmethod
    def addRandomPoints(point_count: int = 1000) -> Points:
        vertices = np.random.uniform(-1, 1, (point_count, 3))
        geometry = Geometry(vertices)
        colors = np.random.uniform(0, 1, (point_count, 4))
        colors[:, 3] = 1.0  # set alpha to 1
        random_points = Points(geometry, color=colors)
        return random_points

    @staticmethod
    def getBunnyPoints() -> Points:
        geometry = MeshUtils.parse_obj_file_manual(os.path.join(models_path, "bunny.obj"))
        geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(geometry.vertices)
        points_bunny = Points(geometry, color=Constants.Color.PURPLE)
        return points_bunny

    @staticmethod
    def getCubePoints() -> Points:
        geometry = MeshUtils.parse_obj_file_manual(os.path.join(models_path, "cube_meshio.obj"))
        geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(geometry.vertices)
        points_bunny = Points(geometry, color=Constants.Color.CYAN)
        return points_bunny

    @staticmethod
    def getSuzannePoints() -> Points:
        geometry = MeshUtils.parse_obj_file_manual(os.path.join(models_path, "suzanne_meshio.obj"))
        geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(geometry.vertices)
        points_bunny = Points(geometry, color=Constants.Color.CYAN)
        return points_bunny

    @staticmethod
    def getHeadPoints() -> Points:
        geometry = MeshUtils.parse_obj_file_manual(os.path.join(models_path, "head_meshio.obj"))
        geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(geometry.vertices)
        points_bunny = Points(geometry, color=Constants.Color.CYAN)
        return points_bunny

    @staticmethod
    def getHeadTexturedMesh() -> TexturedMesh:

        # Load a texture image
        texture_path = os.path.join(images_path, "uv-grid.png")
        texture = Texture.from_file(texture_path)

        # Load a obj model
        obj_path = os.path.join(models_path, "head_meshio.obj")
        # obj_path = os.path.join(models_path, "cube_meshio.obj")
        geometry = MeshUtils.parse_obj_file_manual(obj_path)
        assert geometry.uvs is not None, "The .obj file must contain texture coordinates (vt)"

        # Create a textured mesh
        textured_mesh = TexturedMesh(geometry, texture)
        return textured_mesh
