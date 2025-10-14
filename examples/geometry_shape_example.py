"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
from typing import Sequence

# local imports
from mpl_graph.core import Object3D, Constants
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects import Lines, Polygons
from mpl_graph.geometry import GeometryUtils
from common.geometry_shape import GeometryShape
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

    # Create the scene root
    scene = Object3D()

    # Create a camera
    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[1] = 1.0
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(256, 256, depth_sorting=True)

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add a plane with the texture
    # =============================================================================

    geometry_plane = GeometryShape.plane(1.0, 1.0)
    plane_lines = Lines.from_mesh_geometry(geometry_plane)
    plane_lines.position[0] = -1
    plane_lines.position[1] = 0.7
    scene.add_child(plane_lines)

    geometry_box = GeometryShape.box(1.0, 1.0, 1.0)
    box_polygons = Polygons.from_mesh_geometry(geometry_box)
    box_polygons.material.face_culling = Constants.FaceCulling.FrontSide
    box_polygons.material.face_sorting = True
    box_polygons.position[0] = 1
    box_polygons.position[1] = 0.7
    scene.add_child(box_polygons)

    geometry_grid = GeometryShape.grid(5.0, 5.0)
    grid_lines = Lines.from_mesh_geometry(geometry_grid)
    grid_lines.material.colors = Constants.Color.CYAN
    grid_lines.position[1] = 0
    grid_lines.position[2] = -0.01  # trick to ensure the grid is behind the other objects
    scene.add_child(grid_lines)

    @animation_loop.decorator
    def box_update(delta_time: float) -> Sequence[Object3D]:
        box_polygons.rotation_euler[1] += 0.2 * delta_time
        plane_lines.rotation_euler[1] += 0.2 * delta_time
        # grid_lines.rotation_euler[0] += 0.2 * delta_time

        return [box_polygons, plane_lines, grid_lines]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
