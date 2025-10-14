# stdlib imports
import os
from typing import Sequence
import time

# pip imports
import numpy as np


# local imports
from mpl_graph.core import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects import Polygons
from mpl_graph.core import TransformUtils
from mpl_graph.geometry import Geometry, GeometryUtils
from mpl_graph.core.constants import Constants
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.mesh_utils import MeshUtils
from common.example_utils import ExamplesUtils
from common.geometry_shape import GeometryShape
from common.fps_monitor import FpsMonitor

__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    # camera = CameraPerspective()
    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer, fps=60)

    # =============================================================================
    # Add objects
    # =============================================================================

    # Load a model from an .obj file
    file_path = os.path.join(models_path, "bunny.obj")
    # file_path = os.path.join(models_path, "head.obj")
    # file_path = os.path.join(models_path, "cube.obj")
    # file_path = os.path.join(models_path, "suzanne.obj")

    # parse the .obj file
    mesh_geometry = MeshUtils.parse_obj_file_manual(file_path)

    # Normalize the vertices to fit in a unit cube
    mesh_geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(mesh_geometry.vertices)

    # Create a polygons object
    polygons = Polygons.from_mesh_geometry(mesh_geometry)
    polygons.material.face_culling = Constants.FaceCulling.FrontSide
    polygons.material.face_sorting = True
    polygons.scale[:] = 1
    scene.add_child(polygons)

    fps_monitor = FpsMonitor()

    @animation_loop.decorator
    def polygons_update(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        polygons.rotation_euler[1] += delta_time

        fps_monitor.print_fps()
        # polygons.position[2] = np.cos(present * 3) * 1
        return [polygons]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
