"""
basic example of a camera controller using AWDS keys
"""

# stdlib imports
import os
from typing import Sequence

# pip imports
import numpy as np

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.objects import Scene
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers.renderer import Renderer
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.camera_controller_awds import ObjectControllerWasd
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")
images_path = os.path.join(assets_path, "images")


def main():
    print(f"Controls the object with AWDS keys")
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Scene()

    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add random points
    # =============================================================================

    random_points = SceneExamples.addRandomPoints(1000)
    random_points.scale[:] = 0.5
    scene.add_child(random_points)

    # =============================================================================
    # Add a controller on random_points
    # =============================================================================
    object_controller = ObjectControllerWasd(renderer, random_points)
    object_controller.start()

    @animation_loop.callback_decorator
    def update_controller(delta_time: float) -> Sequence[Object3D]:
        object_controller.update(delta_time)
        return [random_points]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
