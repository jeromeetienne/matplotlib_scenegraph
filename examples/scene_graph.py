# stdlib imports
import os

# pip imports
import numpy as np


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.objects import points
from mpl_graph.objects.points import Points
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.helpers.animation_loop import AnimationLoop
from mpl_graph.helpers.scene_examples import SceneExamples
from mpl_graph.core import Constants


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = RendererMatplotlib(128, 128)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add objects to the scene
    # =============================================================================

    point_object_1 = Points(np.array([[0, 0, 0]]), color=Constants.RED)
    point_object_1.name = "point 1"
    scene.add_child(point_object_1)

    point_object_2 = Points(np.array([[0.0, 0, 0]]), color=Constants.GREEN)
    point_object_2.name = "point 2"
    point_object_2.position[0] = 0.4
    point_object_1.add_child(point_object_2)

    point_object_3 = Points(np.array([[0, 0, 0]]), color=Constants.BLUE)
    point_object_3.name = "point 3"
    point_object_3.position[1] = 0.2
    point_object_2.add_child(point_object_3)

    # =============================================================================
    # Make an animation function
    # =============================================================================

    def animate(delta_time: float, timestamp: float) -> list[Object3D]:
        point_object_1.position[0] = np.cos(timestamp * 2) * 0.5

        point_object_2.position[0] = np.cos(timestamp) * 0.5
        point_object_2.position[1] = np.sin(timestamp) * 0.5

        # point_object_3.position[1] = np.cos(timestamp * 2) * 0.2 + 0.5
        point_object_2.rotation_euler[1] = timestamp * 5

        return [point_object_1, point_object_2, point_object_3]

    animation_loop.add_callback(animate)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
