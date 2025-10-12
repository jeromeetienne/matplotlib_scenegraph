# stdlib imports
import os

# pip imports
import numpy as np
import typing

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.helpers.animation_loop import AnimationLoop
from mpl_graph.helpers.scene_examples import SceneExamples
from mpl_graph.objects.points import Points
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
    renderer = RendererMatplotlib(100, 100)
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

    def post_transform_points(renderer: RendererMatplotlib, camera: CameraBase, vertices_transformed: np.ndarray) -> None:
        
        print(f'Post-transform event: {points.name} with {len(points.vertices)} vertices')

        # Sort points by depth (z value) for proper rendering order
        indices = np.argsort(-vertices_transformed[:, 2])
        vertices_transformed[:] = vertices_transformed[indices]

        # NOTE: Trick to force the static typing of points.vertices/sizes/colors to np.ndarray (and never TransformChain)
        # points.vertices = typing.cast(np.ndarray, points.vertices)

        y_min = vertices_transformed.min(axis=0)
        y_max = vertices_transformed.max(axis=0)
        y_range = y_max - y_min

        for vertex_index, vertex in enumerate(points.vertices):
            color_component = (vertex[1] - y_min)/y_range
            color = (color_component, 0, 1 - color_component, 1)
            points.color[vertex_index] = color

    points.post_transform.connect(post_transform_points)

    def update_points(delta_time: float, timestamp: float) -> list[Object3D]:
        points.rotation_euler[0] = timestamp
        return [points]

    animation_loop.add_callback(update_points)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
