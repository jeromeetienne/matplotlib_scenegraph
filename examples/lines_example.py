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
from mpl_graph.objects.lines import Lines
from mpl_graph.geometry.geometry import Geometry
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(100, 100)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Random animated lines
    # =============================================================================

    lines_count = 10
    vertices = np.random.uniform(-1, 1, size=(lines_count * 2, 3)).astype(np.float32)
    geometry = Geometry(vertices=vertices)
    lines = Lines(geometry)
    scene.add_child(lines)

    @animation_loop.decorator
    def lines_update(delta_time: float) -> Sequence[Object3D]:
        lines.geometry.vertices = np.random.uniform(-1, 1, size=(lines_count * 2, 3)).astype(np.float32)
        return [lines]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
