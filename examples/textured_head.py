"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
import time

# pip imports
import numpy as np

# local imports
from mpl_graph.core import Object3D, Texture
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers import Renderer
from mpl_graph.objects import TexturedMesh
from mpl_graph.geometry import Geometry
from mpl_graph.materials import TextureMeshMaterial
from common.mesh_utils import MeshUtils
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")
images_path = os.path.join(assets_path, "images")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================

    # Create a renderer
    renderer = Renderer(256, 256)

    # Create the scene root
    scene = Object3D()

    # Create a camera and add it to the scene
    # camera = CameraOrthographic()
    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path)
    # remove the alpha channel if any
    texture = texture.strip_alpha() if texture.has_alpha() else texture

    # Load a obj model
    # obj_path = os.path.join(models_path, "head_meshio.obj")
    # obj_path = os.path.join(models_path, "suzanne.obj")
    obj_path = os.path.join(models_path, "cube_meshio.obj")

    mesh_geometry = MeshUtils.parse_obj_file_manual(obj_path)
    assert mesh_geometry.uvs is not None, "The .obj file must contain texture coordinates (vt)"

    # Create a textured mesh
    material = TextureMeshMaterial(texture)
    mesh = TexturedMesh(mesh_geometry, material)
    mesh.scale[:] = 0.5

    # Add the textured mesh to the scene
    scene.add_child(mesh)

    @animation_loop.decorator
    def mesh_update(delta_time: float) -> list[TexturedMesh]:
        present = time.time()
        # mesh.position[0] = np.sin(present)
        mesh.rotation_euler[1] += delta_time
        return [mesh]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    # renderer.render(scene, camera)
    # print("scene rendered")
    # matplotlib.pyplot.show(block=True)

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
