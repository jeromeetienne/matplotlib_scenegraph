"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
import time
from typing import Sequence

# pip imports
import numpy as np

# local imports
from common.scene_examples import SceneExamples
from mpl_graph.core import Object3D, Texture
from mpl_graph.cameras import CameraOrthographic
from mpl_graph.lights.ambient_light import AmbientLight
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Mesh, Scene
from mpl_graph.geometry import Geometry
from mpl_graph.materials import MeshPhongMaterial, MeshTexturedMaterial
from common.controllers.camera_controller_trackball import CameraControllerTrackball
from common.mesh_utils import MeshUtils
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils
from common.geometry_shape import GeometryShape

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
    camera = CameraOrthographic()
    # camera = CameraPerspective()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # Trackball controller bound to this camera
    controller = CameraControllerTrackball(renderer, camera)
    controller.start()

    @animation_loop.event_listener
    def update_camera(_delta: float) -> Sequence[Object3D]:
        has_moved = controller.update(_delta)
        return scene.traverse() if has_moved else []

    # add standard lights
    scene.add(SceneExamples.getThreePointsLighting())

    # =============================================================================
    # Load a model
    # =============================================================================

    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path).ensure_no_alpha()

    # Load a obj geometry
    obj_path = os.path.join(models_path, "suzanne.obj")
    mesh_geometry = MeshUtils.parse_obj_file_manual(obj_path)

    # Create a textured mesh
    material = MeshTexturedMaterial(texture=texture)
    # material = MeshPhongMaterial()
    mesh_textured = Mesh(mesh_geometry, material)
    mesh_textured.rotate_y(np.pi)  # rotate 180deg around Y to have the face looking towards the camera

    # Add the textured mesh to the scene
    scene.add(mesh_textured)

    @animation_loop.event_listener
    def mesh_update(delta_time: float) -> list[Mesh]:
        mesh_textured.rotate_y(0.5 * delta_time)
        return [mesh_textured]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
