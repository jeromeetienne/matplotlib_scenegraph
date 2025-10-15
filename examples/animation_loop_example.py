"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
from typing import Sequence
import time

# pip imports
import numpy as np

# local imports
from mpl_graph.core import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Points, Lines, Scene
from mpl_graph.geometry import Geometry
from mpl_graph.materials import PointsMaterial
from common.example_utils import ExamplesUtils
from common.animation_loop import AnimationLoop


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    # Create a renderer
    renderer = Renderer(512, 512)

    # Create the scene root
    scene = Scene()

    # Create a camera
    camera = CameraOrthographic()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # animation callback with .add_callback()
    # =============================================================================

    # Add points
    point_count = 1000
    geometry = Geometry(np.random.uniform(-1, 1, (point_count, 3)))
    colors = np.array([[1, 0, 0, 1] for i in range(point_count)])
    material = PointsMaterial(colors=colors)
    points = Points(geometry, material)
    points.scale[:] = 0.5
    scene.add(points)

    # add a callback via `animation_loop.add_callback()`
    def update_points(delta_time: float) -> Sequence[Object3D]:
        points.rotate_y(delta_time)

        # return the changed objects thus the renderer can optimize the rendering
        changed_object: Sequence[Points] = [points]
        return changed_object

    animation_loop.add_callback(update_points)

    # =============================================================================
    # animation callback with decorator
    # =============================================================================

    lines_count = 10
    vertices = np.random.uniform(-1, 1, size=(lines_count * 2, 3)).astype(np.float32)
    geometry = Geometry(vertices=vertices)
    lines = Lines(geometry)
    scene.add(lines)

    @animation_loop.event_listener
    def lines_update(delta_time: float) -> Sequence[Object3D]:
        lines.geometry.vertices = np.random.uniform(-1, 1, size=(lines_count * 2, 3)).astype(np.float32)
        return [lines]

    animation_loop.add_callback(lines_update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
