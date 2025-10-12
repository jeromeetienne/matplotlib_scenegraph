"""
basic example of a camera controller using AWDS keys
"""
# stdlib imports
import os

# pip imports
import numpy as np
import matplotlib.pyplot


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.helpers.animation_loop import AnimationLoop
from mpl_graph.helpers.scene_examples import SceneExamples

# local import
from mpl_graph.helpers.camera_controller_awds import CameraController



def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = RendererMatplotlib(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # Add a camera controller
    camera_controller = CameraController(renderer, scene, camera)
    camera_controller.start()

    # =============================================================================
    # Load a model
    # =============================================================================

    random_points = SceneExamples.addRandomPoints(1000)
    random_points.scale[:] = 0.5
    scene.add_child(random_points)

    # def update(delta_time: float, timestamp: float) -> list[Object3D]:
    #     random_points.position[0] = np.cos(timestamp * 5)
    #     random_points.position[1] = np.sin(timestamp * 1.75)
    #     return [random_points]

    # animation_loop.add_callback(update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
