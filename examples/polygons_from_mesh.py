# stdlib imports
import os
from typing import Sequence
import time

# pip imports
import numpy as np


# local imports
from mpl_graph.core import Object3D
from mpl_graph.cameras import CameraPerspective
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects import Polygons, Scene
from mpl_graph.geometry import Geometry, GeometryUtils
from mpl_graph.core.constants import Constants
from common.animation_loop import AnimationLoop
from common.mesh_utils import MeshUtils
from common.example_utils import ExamplesUtils
from common.fps_monitor import FpsMonitor

__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")
output_path = os.path.join(__dirname__, "output")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Scene()

    camera = CameraPerspective()
    # camera = CameraOrthographic()
    scene.add(camera)
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
    file_path = os.path.join(output_path, "box.obj")

    # parse the .obj file
    mesh_geometry = MeshUtils.parse_obj_file_manual(file_path)

    # Normalize the vertices to fit in a unit cube
    mesh_geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(mesh_geometry.vertices)

    # Create a polygons object
    polygons = Polygons.from_mesh_geometry(mesh_geometry)
    polygons.material.face_culling = Constants.FaceCulling.FrontSide
    polygons.material.face_sorting = True
    polygons.scale[:] = 0.5
    scene.add(polygons)

    fps_monitor = FpsMonitor()

    @animation_loop.event_listener
    def polygons_update(delta_time: float) -> Sequence[Object3D]:
        polygons.rotate_x(delta_time)
        polygons.rotate_y(delta_time * 0.5)

        fps_monitor.print_fps()
        return [polygons]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
