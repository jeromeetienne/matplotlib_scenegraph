"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
from typing import Sequence
import time

# pip imports
import os
import numpy as np
import matplotlib.pyplot

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.points import Points
from mpl_graph.geometry.geometry import Geometry
from mpl_graph.core.constants import Constants
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
    camera = CameraPerspective()
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
    colors = np.array([Constants.Color.CYAN for i in range(point_count)])
    points = Points(geometry, color=colors)
    points.scale[:] = 0.5
    scene.add_child(points)

    @animation_loop.decorator
    def update_points(delta_time: float) -> Sequence[Object3D]:
        # Modify the points rotation
        points.rotation_euler[1] += delta_time % (2 * np.pi)
        # return the changed objects thus the renderer can optimize the rendering
        return [points]

    # =============================================================================
    # Render the scene
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
