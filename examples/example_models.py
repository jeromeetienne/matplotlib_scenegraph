"""
basic example of loading and rendering animated 3D models.

Good examples of rendering loop
"""

# pip imports
import time
import numpy as np
import typing

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.objects import Scene
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.example_utils import ExamplesUtils


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Scene()
    scene.name = "Scene"

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(256, 256)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    model_root = Object3D()
    # model_root.scale[:] = 0.1
    scene.add_child(model_root)

    # =============================================================================
    # Add models to the scene
    # =============================================================================

    cube_points = SceneExamples.getCubePoints()
    model_root.add_child(cube_points)
    cube_points.scale[:] = 0.2
    cube_points.position[0] = -3
    cube_points.position[1] = 3

    bunny_points = SceneExamples.getBunnyPoints()
    model_root.add_child(bunny_points)
    bunny_points.scale[:] = 0.2
    bunny_points.position[0] = 3
    bunny_points.position[1] = 3

    head_points = SceneExamples.getHeadPoints()
    model_root.add_child(head_points)
    head_points.scale[:] = 0.2
    head_points.position[0] = 3
    head_points.position[1] = -3

    suzanne_points = SceneExamples.getSuzannePoints()
    model_root.add_child(suzanne_points)
    suzanne_points.scale[:] = 0.2
    suzanne_points.position[0] = -3
    suzanne_points.position[1] = -3

    @animation_loop.callback_decorator
    def update_model_root(delta_time: float) -> typing.Sequence[Object3D]:
        present = time.time()
        range = np.sin(present) * 1 + 2
        bunny_points.position[1] = np.abs(np.cos(present * 5) * range)
        return [bunny_points]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
