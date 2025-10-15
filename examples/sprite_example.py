"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
from typing import Sequence
import time

# pip imports
import numpy as np

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.core.texture import Texture
from mpl_graph.objects import Sprite, Points, Scene
from mpl_graph.geometry.geometry import Geometry
from mpl_graph.materials import PointsMaterial, SpriteMaterial
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(data_path, "models")
images_path = os.path.join(data_path, "images")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Scene()

    camera = CameraOrthographic()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer()
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add objects to the scene
    # =============================================================================

    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path)

    # Create a sprite
    material1 = SpriteMaterial(texture=texture)
    sprite1 = Sprite(material1)
    sprite1.scale[:] = 2
    scene.add(sprite1)

    # Create a sprite
    material2 = SpriteMaterial(texture=texture)
    sprite2 = Sprite(material2)
    sprite2.scale[:] = 2
    scene.add(sprite2)

    @animation_loop.event_listener
    def on_update(delta_time: float) -> Sequence[Object3D]:
        angle1 = time.time()
        sprite1.position[1] = np.cos(angle1) * 0.2
        sprite1.position[2] = np.sin(angle1) * 3

        angle2 = angle1 + np.pi
        sprite2.position[1] = np.cos(angle2) * 0.2
        sprite2.position[2] = np.sin(angle2) * 3
        return [sprite1, sprite2]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
