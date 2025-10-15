"""
example of using renderer events to modify point colors and sizes based on depth
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
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils
from mpl_graph.objects import Points, Scene
from mpl_graph.geometry import Geometry
from mpl_graph.materials import PointsMaterial
from mpl_graph.cameras.camera import Camera


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Scene()

    camera = CameraOrthographic()
    scene.add(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(256, 256)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    point_count = 100
    vertices = np.random.uniform(-0.5, 0.5, (point_count, 3))
    geometry = Geometry(vertices=vertices)
    colors = np.array([[1.0, 0.0, 0.0, 1.0] for i in range(point_count)])
    sizes = np.array([100.0 for i in range(point_count)])
    edge_colors = np.array([[0.0, 0.0, 0.0, 0.2] for i in range(point_count)])
    material = PointsMaterial(colors=colors, sizes=sizes, edge_colors=edge_colors)
    points = Points(geometry, material)
    scene.add(points)

    @points.post_transform.event_listener
    def post_transform_points(vertices_transformed: np.ndarray) -> None:

        # sort inplace transformed positions by z value (3rd column). Largest z first
        sorted_indices = np.argsort(vertices_transformed[:, 2])
        vertices_transformed[:] = vertices_transformed[sorted_indices]
        # apply same sorting to points.geometry.vertices, points.colors etc... to preserve the correct association
        points.geometry.vertices[:] = points.geometry.vertices[sorted_indices]
        points.material.colors[:] = points.material.colors[sorted_indices]
        points.material.sizes[:] = points.material.sizes[sorted_indices]
        points.material.edge_colors[:] = points.material.edge_colors[sorted_indices]
        # points.edge_widths[:] = points.edge_widths[sorted_indices]

        # get the min and max y values
        z_min = vertices_transformed[:, 2].min()
        z_max = vertices_transformed[:, 2].max()
        z_range = z_max - z_min

        for vertex_index, vertex in enumerate(vertices_transformed):
            color_component = (vertex[2] - z_min) / z_range
            color = np.array([1.0, 1.0 - color_component, 1.0 - color_component, 1.0], dtype=np.float32)
            points.material.colors[vertex_index] = color

    def update_points(delta_time: float) -> Sequence[Object3D]:
        points.rotate_x(0.5 * delta_time)
        return [points]

    animation_loop.add_callback(update_points)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
