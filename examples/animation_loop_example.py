"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
from typing import Sequence
import time

# pip imports
import numpy as np

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.points import Points
from mpl_graph.objects.lines import Lines
from mpl_graph.core.geometry import Geometry
from common.example_utils import ExamplesUtils
from common.animation_loop import AnimationLoop


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    # Create a renderer
    renderer = Renderer(512, 512)

    # Create the scene root
    scene = Object3D()

    # Create a camera
    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Build your scene
    # =============================================================================

    # Add points
    point_count = 1000
    geometry = Geometry(np.random.uniform(-1, 1, (point_count, 3)))
    colors = np.array([[1, 0, 0, 1] for i in range(point_count)])
    points = Points(geometry, color=colors)
    points.scale[:] = 0.5
    scene.add_child(points)

    # add a callback via `animation_loop.add_callback()`
    def update_points(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        points.rotation_euler[1] = present % (2 * np.pi)

        # return the changed objects thus the renderer can optimize the rendering
        changed_object: Sequence[Points] = [points]
        return changed_object

    animation_loop.add_callback(update_points)

    # =============================================================================
    # Add some lines
    # =============================================================================

    lines_count = 10
    vertices = np.random.uniform(-1, 1, size=(lines_count * 2, 3)).astype(np.float32)
    geometry = Geometry(vertices=vertices)
    lines = Lines(geometry)
    scene.add_child(lines)

    @animation_loop.decorator
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
