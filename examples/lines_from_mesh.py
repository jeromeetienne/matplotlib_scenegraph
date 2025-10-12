# stdlib imports
import os
from typing import Sequence

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
from common.mesh_parser_obj_manual import MeshParserObjManual
from mpl_graph.core.geometry import Geometry


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
    renderer = Renderer(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Random animated lines
    # =============================================================================

    if True:
        # Load a model from an .obj file
        # file_path = os.path.join(models_path, "cube_meshio.obj")
        file_path = os.path.join(models_path, "suzanne_meshio.obj")

        lines = SceneExamples.lines_from_obj(file_path)
        lines.scale[:] = 0.5
        scene.add_child(lines)

        def lines_update(delta_time: float, time_stamp: float) -> Sequence[Object3D]:
            lines.rotation_euler[1] = time_stamp
            lines.position[1] = np.cos(time_stamp * 3) * 1
            return [lines]

        animation_loop.add_callback(lines_update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
