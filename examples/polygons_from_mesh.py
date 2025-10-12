# stdlib imports
import os
from typing import Sequence

# pip imports
import numpy as np


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.polygons import Polygons
from mpl_graph.core.geometry import Geometry
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from examples.common.mesh_utils import MeshUtils


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

    if True:
        # Load a model from an .obj file
        # file_path = os.path.join(models_path, "cube_meshio.obj")
        file_path = os.path.join(models_path, "suzanne_meshio.obj")

        polygons = SceneExamples.polygons_from_obj(file_path)
        polygons.scale[:] = 0.5
        scene.add_child(polygons)

        def polygons_update(delta_time: float, time_stamp: float) -> Sequence[Object3D]:
            polygons.rotation_euler[1] = time_stamp
            polygons.position[1] = np.cos(time_stamp * 3) * 1
            return [polygons]

        animation_loop.add_callback(polygons_update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
