"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
from typing import Sequence

# pip imports
import numpy as np

# local imports
from mpl_graph.cameras import CameraOrthographic, CameraPerspective
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Mesh, Scene
from mpl_graph.materials import MeshNormalMaterial
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
    scene = Scene()

    # Create a camera and add it to the scene
    # camera = CameraOrthographic()
    camera = CameraPerspective()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    # Load a obj geometry
    obj_path = os.path.join(models_path, "head.obj")
    mesh_geometry = MeshUtils.parse_obj_file_manual(obj_path)

    # Create a mesh
    material = MeshNormalMaterial()
    mesh = Mesh(mesh_geometry, material)
    mesh.rotate_y(np.pi)  # rotate 180deg around Y to have the face looking towards the camera

    # Add the textured mesh to the scene
    scene.add(mesh)

    # update to rotate the mesh
    @animation_loop.event_listener
    def mesh_update(delta_time: float) -> list[Mesh]:
        mesh.rotate_y(0.5 * delta_time)
        return [mesh]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
