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
from mpl_graph.objects.lines import Lines
from mpl_graph.geometry.geometry import Geometry
from mpl_graph.core.constants import Constants
from mpl_graph.objects import Polygons
from common.geometry_shape import GeometryShape
from common.example_utils import ExamplesUtils
from common.animation_loop import AnimationLoop


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================

    # Create a renderer
    renderer = Renderer(100, 100)

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

    geometry_box1 = GeometryShape.box()
    box1_lines = Lines.from_mesh_geometry(geometry_box1)
    box1_lines.position[0] = -0.5
    box1_lines.color = Constants.Color.GREEN
    scene.add_child(box1_lines)

    geometry_box2 = GeometryShape.box()
    box2_lines = Lines.from_mesh_geometry(geometry_box2)
    box2_lines.position[0] = 0.5
    box2_lines.color = Constants.Color.RED
    scene.add_child(box2_lines)

    @animation_loop.decorator
    def update_points(delta_time: float) -> Sequence[Object3D]:
        box1_lines.position[0] = np.cos(time.time()) * 1
        box1_lines.position[2] = np.sin(time.time()) * 1

        box2_lines.position[0] = np.cos(time.time() + np.pi) * 1
        box2_lines.position[2] = np.sin(time.time() + np.pi) * 1

        return [box1_lines, box2_lines]

    # =============================================================================
    # Render the scene
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
