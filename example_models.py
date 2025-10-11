# stdlib imports
import os

# pip imports
import numpy as np


# local imports
from core.object_3d import Object3D
from cameras.camera_orthographic import CameraOrthographic
from renderers.matplotlib.renderer import RendererMatplotlib
from helpers.animation_loop import AnimationLoop
from helpers.scene_examples import SceneExamples


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()
    scene.name = "Scene"

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = RendererMatplotlib()
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    model_root = Object3D()
    # model_root.scale[:] = 0.1
    scene.add_child(model_root)

    SceneExamples.addAnimatedModels(model_root, animation_loop)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
