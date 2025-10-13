"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
from typing import Sequence
import time

# pip imports
import numpy as np

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.textured_mesh import TexturedMesh
from mpl_graph.core.texture import Texture
from mpl_graph.objects.sprite import Sprite
from mpl_graph.objects.points import Points
from mpl_graph.core.geometry import Geometry
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(data_path, "models")
images_path = os.path.join(data_path, "images")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(100, 100)
    # Create an animation loop
    video_path = os.path.join(__dirname__, "output/sprite_animation.mp4")
    video_duration = 10.0  # seconds
    print(f"Animation will be saved to: {video_path}")
    animation_loop = AnimationLoop(renderer, video_duration=video_duration, video_path=video_path)
    # TODO how can i stop whenever the video is done

    # =============================================================================
    # Add a point cloud
    # =============================================================================

    point_count = 10
    vertices = np.random.uniform(-1, 1, (point_count, 3))
    geometry = Geometry(vertices)
    colors = np.array([[1, 0, 0, 1] for i in range(point_count)])
    points = Points(geometry, color=colors)
    points.scale[:] = 0.5
    scene.add_child(points)

    @animation_loop.decorator
    def points_animation(delta_time: float) -> Sequence[Points]:
        present = time.time()
        points.position[0] = np.sin(present * 3) * 0.5
        points.position[1] = np.cos(present * 3) * 0.5

        points.scale[0] = 0.5 + 0.1 * np.cos(present * 2.0)
        points.scale[1] = 0.5 + 0.1 * np.sin(present * 2.0)

        return [points]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
