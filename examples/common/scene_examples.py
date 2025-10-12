# stdlib imports
import os

# pip imports
import numpy as np
import matplotlib.image

# local imports
from mpl_graph.objects.points import Points
from mpl_graph.core.object_3d import Object3D
from mpl_graph.core.constants import Constants
from mpl_graph.core.transform_utils import TransformUtils
from mpl_graph.objects.textured_mesh import TexturedMesh
from mpl_graph.objects.polygons import Polygons
from mpl_graph.core.geometry import Geometry
from mpl_graph.core.texture import Texture
from .geometry_shape import GeometryShape

from .mesh_parser_obj_manual import MeshParserObjManual


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(__dirname__, "../../assets")
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

        geometry = Geometry(vertices)

        polygons = Polygons(geometry, polygon_count, vertices_per_polygon)

        # polygons = Points(vertices)
        return polygons

    @staticmethod
    def addRandomPoints(point_count: int = 1000) -> Points:
        vertices = np.random.uniform(-1, 1, (point_count, 3))
        colors = np.random.uniform(0, 1, (point_count, 4))
        colors[:, 3] = 1.0  # set alpha to 1
        random_points = Points(vertices, color=colors)
        return random_points

    @staticmethod
    def getBunnyPoints() -> Points:
        face_indices, vertex_coords, uv_coords, normal_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "bunny.obj"))
        vertex_coords = TransformUtils.normalize_vertices_to_unit_cube(vertex_coords)
        points_bunny = Points(vertex_coords, color=Constants.Color.PURPLE)
        return points_bunny

    @staticmethod
    def getCubePoints() -> Points:
        face_indices, vertex_coords, uv_coords, normal_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "cube_meshio.obj"))
        vertex_coords = TransformUtils.normalize_vertices_to_unit_cube(vertex_coords)
        points_bunny = Points(vertex_coords, color=Constants.Color.CYAN)
        return points_bunny

    @staticmethod
    def getSuzannePoints() -> Points:
        faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "suzanne_meshio.obj"))
        vertices_coords = TransformUtils.normalize_vertices_to_unit_cube(vertices_coords)
        points_bunny = Points(vertices_coords, color=Constants.Color.CYAN)
        return points_bunny

    @staticmethod
    def getHeadPoints() -> Points:
        faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(os.path.join(models_path, "head_meshio.obj"))
        vertices_coords = TransformUtils.normalize_vertices_to_unit_cube(vertices_coords)
        points_bunny = Points(vertices_coords, color=Constants.Color.CYAN)
        return points_bunny

    @staticmethod
    def getHeadTexturedMesh() -> TexturedMesh:

        # Load a texture image
        texture_path = os.path.join(images_path, "uv-grid.png")
        texture = Texture.from_file(texture_path)

        # Load a obj model
        obj_path = os.path.join(models_path, "head_meshio.obj")
        # obj_path = os.path.join(models_path, "cube_meshio.obj")
        faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(obj_path)
        assert uvs_coords is not None, "The .obj file must contain texture coordinates (vt)"

        geometry = Geometry(vertices_coords, faces_indices, uvs_coords, normals_coords)

        # Create a textured mesh
        textured_mesh = TexturedMesh(geometry, texture)
        return textured_mesh
