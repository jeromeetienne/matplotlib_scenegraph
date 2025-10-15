"""
Example: Control the camera with a trackball (orbit/pan/dolly).
Left drag = orbit, Middle drag or Shift+Left = pan, Right drag or Ctrl+Left = dolly, Scroll = zoom
"""

import os
from typing import Sequence
import numpy as np

from mpl_graph.core import Object3D
from mpl_graph.objects import Scene
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers import Renderer
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.controllers.camera_controller_trackball import CameraControllerTrackball
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))


def main():
    print("Controls: Left drag orbit | Middle drag/Shift+Left pan | Right drag/Ctrl+Left dolly | Wheel zoom")

    # =============================================================================
    # Set up the scene
    # =============================================================================

    # Create a renderer
    renderer = Renderer(512, 512)

    # Create the scene root
    scene = Scene()

    # Create a camera
    camera = CameraPerspective()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Init camera controller
    # =============================================================================

    # Trackball controller bound to this camera
    controller = CameraControllerTrackball(renderer, camera)
    controller.start()

    @animation_loop.event_listener
    def update_camera(time_delta: float) -> Sequence[Object3D]:
        has_moved = controller.update(time_delta)
        return scene.traverse() if has_moved else []

    # =============================================================================
    # Add some content
    # =============================================================================
    # Simple content
    points = SceneExamples.addRandomPoints(1000)
    scene.add(points)

    # =============================================================================
    # Start animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
