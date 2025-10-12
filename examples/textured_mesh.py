"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os

# pip imports
import matplotlib.image
import matplotlib.pyplot
import numpy as np

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.objects.textured_mesh import TexturedMesh
from mpl_graph.core.texture import Texture
from mpl_graph.objects.points import Points
from common.mesh_parser_obj_manual import MeshParserObjManual
from common.animation_loop import AnimationLoop

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(data_path, "models")
images_path = os.path.join(data_path, "images")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = RendererMatplotlib(256, 256)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # # =============================================================================
    # # Add a point cloud
    # # =============================================================================

    # point_count = 1000
    # vertices = np.random.uniform(-1, 1, (point_count, 3))
    # colors = np.array([[1, 0, 0, 1] for i in range(point_count)])
    # points = Points(vertices, color=colors)
    # points.scale[:] = 0.5
    # scene.add_child(points)

    # def update(delta_time: float, timestamp: float) -> list[Object3D]:
    #     points.position[0] = np.cos(timestamp * 5)
    #     points.position[1] = np.sin(timestamp * 1.75)
    #     return [points]

    # animation_loop.add_callback(update)

    # =============================================================================
    # Load a model
    # =============================================================================

    # def __init__(self, faces_indices: np.ndarray, vertices_coords: np.ndarray, uvs_coords: np.ndarray, texture: np.ndarray):

    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path)
    texture = texture.strip_alpha() if texture.has_alpha() else texture

    # Load a textured mesh from an .obj file
    obj_path = os.path.join(models_path, "head_meshio.obj")
    # obj_path = os.path.join(models_path, "cube_meshio.obj")
    faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(obj_path)
    assert uvs_coords is not None, "The .obj file must contain texture coordinates (vt)"
    textured_mesh = TexturedMesh(faces_indices, vertices_coords, uvs_coords, texture.data)
    textured_mesh.name = "TexturedMesh"

    scene.add_child(textured_mesh)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    renderer.render(scene, camera)
    print("scene rendered")
    matplotlib.pyplot.show(block=True)


if __name__ == "__main__":
    main()
