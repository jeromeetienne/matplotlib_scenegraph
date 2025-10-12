"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os

# pip imports
import matplotlib.image
import matplotlib.pyplot


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.objects.textured_mesh import TexturedMesh
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
    renderer = RendererMatplotlib(100, 100)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    # def __init__(self, faces_indices: np.ndarray, vertices_coords: np.ndarray, uvs_coords: np.ndarray, texture: np.ndarray):

    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = matplotlib.image.imread(texture_path)

    # Load a textured mesh from an .obj file
    obj_path = os.path.join(models_path, "head_meshio.obj")
    faces_indices, vertices_coords, uvs_coords, normals_coords = MeshParserObjManual.parse_obj_file(obj_path)
    assert uvs_coords is not None, "The .obj file must contain texture coordinates (vt)"
    textured_mesh = TexturedMesh(faces_indices, vertices_coords, uvs_coords, texture)
    textured_mesh.name = "TexturedMesh"

    scene.add_child(textured_mesh)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    renderer.render(scene, camera)

    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()
