"""
basic example of rendering a rotating point cloud
"""

# pip imports
import os
import numpy as np
import matplotlib.pyplot

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.points import Points
from mpl_graph.core.geometry import Geometry
from common.example_utils import ExamplesUtils


def main():
    # Create the scene root
    scene = Object3D()

    # Create a camera
    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(512, 512)

    # Add points
    point_count = 1000
    geometry = Geometry(np.random.uniform(-1, 1, (point_count, 3)))
    colors = np.array([[1, 0, 0, 1] for i in range(point_count)])
    points = Points(geometry, color=colors)
    points.scale[:] = 0.5
    scene.add_child(points)

    # Render the scene
    renderer.render(scene, camera)

    # if we are in testing mode, exit now
    if ExamplesUtils.postamble():
        return

    # show the result
    matplotlib.pyplot.show(block=True)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
