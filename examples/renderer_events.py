"""
example of using renderer events to modify point colors and sizes based on depth
"""

# stdlib imports
from typing import Sequence

# pip imports
import numpy as np

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils
from mpl_graph.objects.points import Points
from mpl_graph.core.geometry import Geometry
from mpl_graph.cameras.camera_base import CameraBase


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

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

    point_count = 100
    vertices = np.random.uniform(-0.5, 0.5, (point_count, 3))
    geometry = Geometry(vertices=vertices)
    colors = np.array([[1.0, 0.0, 0.0, 1.0] for i in range(point_count)])
    sizes = np.array([100.0 for i in range(point_count)])
    edge_colors = np.array([[0.0, 0.0, 0.0, 0.2] for i in range(point_count)])
    points = Points(geometry, color=colors, sizes=sizes, edge_colors=edge_colors)
    scene.add_child(points)

    def post_transform_points(renderer: Renderer, camera: CameraBase, vertices_transformed: np.ndarray) -> None:

        # sort inplace transformed positions by z value (3rd column). Largest z first
        sorted_indices = np.argsort(vertices_transformed[:, 2])
        vertices_transformed[:] = vertices_transformed[sorted_indices]
        # apply same sorting to points.geometry.vertices, points.colors etc... to preserve the correct association
        points.geometry.vertices[:] = points.geometry.vertices[sorted_indices]
        points.colors[:] = points.colors[sorted_indices]
        points.sizes[:] = points.sizes[sorted_indices]
        points.edge_colors[:] = points.edge_colors[sorted_indices]
        # points.edge_widths[:] = points.edge_widths[sorted_indices]

        # get the min and max y values
        z_min = vertices_transformed[:, 2].min()
        z_max = vertices_transformed[:, 2].max()
        z_range = z_max - z_min

        for vertex_index, vertex in enumerate(vertices_transformed):
            color_component = (vertex[2] - z_min) / z_range
            color = np.array([1.0, 1.0 - color_component, 1.0 - color_component, 1.0], dtype=np.float32)
            points.colors[vertex_index] = color

    points.post_transform.subscribe(post_transform_points)

    def update_points(delta_time: float, timestamp: float) -> Sequence[Object3D]:
        points.rotation_euler[0] = timestamp
        return [points]

    animation_loop.add_callback(update_points)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
