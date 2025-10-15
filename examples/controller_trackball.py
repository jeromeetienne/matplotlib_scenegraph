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
from examples.common.controllers.object_controller_trackball import CameraControllerTrackball
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))


def main():
    print("Controls: Left drag orbit | Middle drag/Shift+Left pan | Right drag/Ctrl+Left dolly | Wheel zoom")

    # Scene & camera
    scene = Scene()
    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0
    # camera.position[:] = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    # Renderer
    renderer = Renderer(512, 512)

    # Animation loop
    animation_loop = AnimationLoop(renderer)

    # Trackball controller bound to this camera
    controller = CameraControllerTrackball(renderer, camera, target=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    controller.start()

    @animation_loop.decorator_callback
    def update_camera(_delta: float) -> Sequence[Object3D]:
        has_moved = controller.update(_delta)
        return scene.traverse() if has_moved else []

    # Simple content
    points = SceneExamples.addRandomPoints(1000)
    scene.add_child(points)

    # run
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
