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
from mpl_graph.objects.polygons import Polygons
from mpl_graph.geometry.geometry import Geometry
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.mesh_utils import MeshUtils
from common.example_utils import ExamplesUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")


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
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add objects
    # =============================================================================

    # Create a list of polygons, each polygon is a list of (x,y) points
    # Add a z=0 to each (x, y) point to make (x, y, z)
    vertices = (
        np.array(
            [
                [(1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)],
                [(3, 1, 0), (4, 1, 0), (4, 2, 0), (3, 2, 0)],
                [(1, 3, 0), (2, 3, 0), (2, 4, 0), (1, 4, 0)],
            ],
            dtype=np.float32,
        )
        / 5
    )

    polygon_count = vertices.shape[0]
    vertices_per_polygon = vertices.shape[1]
    vertices = vertices.reshape(polygon_count * vertices_per_polygon, 3)
    geometry = Geometry(vertices=vertices)
    quad_polygons = Polygons(geometry, polygon_count, vertices_per_polygon)
    scene.add_child(quad_polygons)

    @animation_loop.decorator
    def quad_polygons_update(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        quad_polygons.position[0] = np.cos(present) * 0.5
        return [quad_polygons]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
