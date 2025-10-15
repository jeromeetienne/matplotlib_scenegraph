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
from mpl_graph.core import Object3D, Texture
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.core.constants import Constants
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Mesh, Scene
from mpl_graph.lights import DirectionalLight, PointLight, AmbientLight, Light
from mpl_graph.geometry import Geometry
from mpl_graph.materials import MeshPhongMaterial, MeshBasicMaterial, MeshNormalMaterial, MeshDepthMaterial, MeshTexturedMaterial
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
    renderer = Renderer(512, 512, background_color=Constants.Color.LIGHT_GRAY)

    # Create the scene root
    scene = Scene()

    # Create a camera and add it to the scene
    camera = CameraOrthographic()
    camera.near = 0.1
    camera.far = 4
    # camera = CameraPerspective()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    # Create a directional key light
    directional_light_key = DirectionalLight(intensity=1, color=Constants.Color.LIGHT_BLUE)
    directional_light_key.position = np.array((1.0, 0.0, -1.0))
    scene.add(directional_light_key)
    scene.add(directional_light_key.target)

    # Create a directional fill light
    directional_light_fill = DirectionalLight(color=Constants.Color.WHITE, intensity=0.3)
    directional_light_fill.position = np.array((-1.0, 0.0, -1.0))
    scene.add(directional_light_fill)
    scene.add(directional_light_fill.target)

    # Create a directional backlight light
    directional_light_backlight = DirectionalLight(color=Constants.Color.WHITE, intensity=0.2)
    directional_light_backlight.position = np.array((0.0, 1.0, -1.0))
    scene.add(directional_light_backlight)
    scene.add(directional_light_backlight.target)

    # add a ambient light
    ambient_light = AmbientLight(color=Constants.Color.WHITE, intensity=0.2)
    scene.add(ambient_light)

    @animation_loop.event_listener
    def light_update(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        directional_light_key.position[0] = 2.0 * np.cos(present)
        directional_light_key.position[2] = 2.0 * np.sin(present)
        return [mesh, directional_light_key]

    # =============================================================================
    #
    # =============================================================================

    # Load a obj geometry
    obj_path = os.path.join(models_path, "head.obj")
    mesh_geometry = MeshUtils.parse_obj_file_manual(obj_path)

    # Create a mesh
    material = MeshPhongMaterial(shininess=30, color=Constants.Color.LIGHT_BLUE)
    mesh = Mesh(mesh_geometry, material)
    scene.add(mesh)
    mesh.scale[:] = 1
    mesh.rotate_y(np.pi)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
