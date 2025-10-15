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
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects import Scene, Text
from mpl_graph.materials import TextMaterial
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils
from mpl_graph.core import Constants

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
    renderer = Renderer(100, 100, background_color=Constants.Color.LIGHT_GRAY)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    material = TextMaterial()
    text = Text("hello world", material)
    text.scale[:] = 0.5
    scene.add(text)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
