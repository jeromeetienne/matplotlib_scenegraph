"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os

# pip imports
import numpy as np


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.helpers.animation_loop import AnimationLoop
from mpl_graph.helpers.scene_examples import SceneExamples
from mpl_graph.objects.points import Points

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

    # =============================================================================
    # Load a model
    # =============================================================================

    point_count = 1000
    vertices = np.random.uniform(-1, 1, (point_count, 3))
    colors = np.array([[1, 0, 0, 1] for i in range(point_count)])
    points = Points(vertices, color=colors)
    points.scale[:] = 0.5
    scene.add_child(points)

    def update(delta_time: float, timestamp: float) -> list[Object3D]:
        points.position[0] = np.cos(timestamp * 5)
        points.position[1] = np.sin(timestamp * 1.75)
        return [points]

    animation_loop.add_callback(update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
