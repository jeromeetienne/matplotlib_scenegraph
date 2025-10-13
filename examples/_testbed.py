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
from common.animation_loop import AnimationLoop
from mpl_graph.objects.lines import Lines
from common.scene_examples import SceneExamples
from mpl_graph.objects.polygons import Polygons
from common.mesh_utils import MeshUtils


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
    # Load a model
    # =============================================================================

    if False:
        head_points = SceneExamples.getHeadPoints()
        scene.add_child(head_points)

        def head_points_update(delta_time: float, time_stamp: float) -> Sequence[Object3D]:
            head_points.position[0] = np.cos(time_stamp) * 0.5
            head_points.scale[:] = np.cos(time_stamp) * 0.5 + 1.5
            return [head_points]

        animation_loop.add_callback(head_points_update)

    # =============================================================================
    # Random animated lines
    # =============================================================================

    if True:
        # Load a model from an .obj file
        # file_path = os.path.join(models_path, "cube.obj")
        file_path = os.path.join(models_path, "suzanne_meshio.obj")
        # file_path = os.path.join(models_path, "head_meshio.obj")

        polygons = SceneExamples.polygons_from_obj(file_path)
        polygons.scale[:] = 0.5
        scene.add_child(polygons)

        def polygons_update(delta_time: float) -> Sequence[Object3D]:
            present = time.time()
            polygons.rotation_euler[1] = present
            polygons.position[1] = np.cos(present * 3) * 1
            return [polygons]

        animation_loop.add_callback(polygons_update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
