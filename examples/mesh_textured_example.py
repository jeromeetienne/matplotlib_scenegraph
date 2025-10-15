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
from mpl_graph.core import Texture, Constants, Object3D
from mpl_graph.cameras import CameraOrthographic
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Mesh, Scene
from mpl_graph.materials import MeshTexturedMaterial, MeshPhongMaterial
from mpl_graph.lights import DirectionalLight, AmbientLight, Light
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

    # =============================================================================
    # Add a lighting
    # =============================================================================

    # add a ambient light
    ambient_light = AmbientLight(intensity=0.2)
    scene.add(ambient_light)

    # Create a directional key light
    directional_light = DirectionalLight(intensity=0.8)
    directional_light.position = np.array((1.0, 0.0, -1.0))
    scene.add(directional_light)
    scene.add(directional_light.target)

    @animation_loop.event_listener
    def light_update(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        directional_light.position[0] = 2.0 * np.cos(present)
        directional_light.position[2] = 2.0 * np.sin(present)
        return [mesh, directional_light]

    # =============================================================================
    # Load a model
    # =============================================================================

    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path).ensure_no_alpha()

    # Load a obj model
    obj_path = os.path.join(models_path, "head.obj")
    # obj_path = os.path.join(models_path, "suzanne.obj")
    # obj_path = os.path.join(models_path, "box.obj")
    mesh_geometry = MeshUtils.parse_obj_file_manual(obj_path)

    # Create a textured mesh
    material = MeshTexturedMaterial(texture=texture, shininess=120.0)
    mesh = Mesh(mesh_geometry, material)
    mesh.rotate_y(np.pi)  # rotate the model to face the camera
    scene.add(mesh)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
