"""
basic example of loading and rendering animated 3D models.

Good examples of rendering loop
"""

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.helpers.animation_loop import AnimationLoop
from mpl_graph.helpers.scene_examples import SceneExamples


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
    renderer = RendererMatplotlib(256, 256)
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
