# stdlib imports
import os
from time import time
from typing import Sequence

# pip imports
import numpy as np


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.objects import points
from mpl_graph.objects.points import Points
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.core import Constants
from mpl_graph.core.geometry import Geometry
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.example_utils import ExamplesUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    # camera = CameraOrthographic()
    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add objects to the scene
    # =============================================================================

    point_object_1 = Points(Geometry(np.array([[0, 0, 0]])), color=Constants.Color.RED)
    point_object_1.name = "point 1"
    scene.add_child(point_object_1)

    point_object_2 = Points(Geometry(np.array([[0.0, 0, 0]])), color=Constants.Color.GREEN)
    point_object_2.name = "point 2"
    point_object_2.position[0] = 0.4
    point_object_1.add_child(point_object_2)

    point_object_3 = Points(Geometry(np.array([[0, 0, 0]])), color=Constants.Color.BLUE)
    point_object_3.name = "point 3"
    point_object_3.position[1] = 0.2
    point_object_2.add_child(point_object_3)

    # =============================================================================
    # Make an animation function
    # =============================================================================

    def animate(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        # point_object_1.rotation_euler[2] = present * 2
        point_object_1.position[2] = np.cos(present * 10) * 4
        point_object_1.scale[:] = 1.5

        # point_object_2.position[0] = np.cos(present) * 0.5
        # point_object_2.position[1] = np.sin(present) * 0.5
        # point_object_2.rotation_euler[2] = present * 5
        point_object_2.scale[:] = 0.5

        # point_object_3.position[1] = np.cos(present * 2) * 0.2 + 0.2
        # point_object_2.rotation_euler[1] = present * 5

        return [point_object_1, point_object_2, point_object_3]

    animation_loop.add_callback(animate)

    # =============================================================================
    # Lines
    # =============================================================================
    file_path = os.path.join(models_path, "suzanne.obj")

    lines = SceneExamples.lines_from_obj(file_path)
    lines.scale[:] = 0.5
    scene.add_child(lines)

    # @animation_loop.decorator
    def lines_update(delta_time: float) -> Sequence[Object3D]:
        time_stamp = time.time()
        lines.rotation_euler[1] = time_stamp
        lines.position[2] = np.cos(time_stamp * 3) * 1
        return [lines]

    animation_loop.add_callback(lines_update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
