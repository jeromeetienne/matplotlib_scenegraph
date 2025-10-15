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

    # Load a model geometry
    file_path = os.path.join(models_path, "bunny.obj")
    mesh_geometry = MeshUtils.parse_obj_file_manual(file_path)

    # Fit the vertices to a unit cube
    mesh_geometry.vertices = GeometryUtils.fit_unit_cube(mesh_geometry.vertices)

    # Create a polygons object
    polygons1 = Polygons.from_mesh_geometry(mesh_geometry.copy())
    polygons1.material.face_culling = Constants.FaceCulling.FrontSide
    polygons1.material.depth_sorting = True
    polygons1.scale[:] = 0.5
    scene.add(polygons1)

    # Create a polygons object
    polygons2 = Polygons.from_mesh_geometry(mesh_geometry.copy())
    polygons2.material.face_culling = Constants.FaceCulling.FrontSide
    polygons2.material.depth_sorting = True
    polygons2.scale[:] = 0.5
    scene.add(polygons2)

    fps_monitor = FpsMonitor()

    @animation_loop.event_listener
    def polygons_update(delta_time: float) -> Sequence[Object3D]:
        polygons1.rotate_x(delta_time)
        polygons1.rotate_y(delta_time / 2)

        angle1 = time.time()
        polygons1.position[1] = np.cos(angle1) * 0.2
        polygons1.position[2] = np.sin(angle1) * 3

        angle2 = angle1 + np.pi
        polygons2.position[1] = np.cos(angle2) * 0.2
        polygons2.position[2] = np.sin(angle2) * 3

        fps_monitor.print_fps()
        return [polygons1, polygons2]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
