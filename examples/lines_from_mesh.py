# stdlib imports
import os
import time
from typing import Sequence

# pip imports
import numpy as np


# local imports
from mpl_graph.core import Object3D, Constants
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Lines, Scene
from mpl_graph.materials import LinesMaterial
from common.geometry_shape import GeometryShape
from common.example_utils import ExamplesUtils
from common.animation_loop import AnimationLoop
from common.mesh_utils import MeshUtils
from mpl_graph.geometry import GeometryUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(__dirname__, "output")
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Scene()

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

    # Load a model from an .obj file
    # file_path = os.path.join(models_path, "cube_meshio.obj")
    file_path = os.path.join(models_path, "suzanne.obj")
    file_path = os.path.join(output_path, "box.obj")

    # parse the .obj file
    mesh_geometry = MeshUtils.parse_obj_file_manual(file_path)
    assert mesh_geometry.indices is not None, "The .obj file must contain face indices"

    # Normalize the vertices to fit in a unit cube
    mesh_geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(mesh_geometry.vertices)

    # Build the lines object
    lines = Lines.from_mesh_geometry(mesh_geometry, dedup_edges=True)
    lines.material.colors = Constants.Color.CYAN
    lines.scale[:] = 0.5
    scene.add_child(lines)

    @animation_loop.event_listener
    def lines_update(delta_time: float) -> Sequence[Object3D]:
        lines.rotate_x(delta_time / 2)
        lines.rotate_y(delta_time)
        return [lines]

    # Compare with and without deduplication of edges
    lines_without_dedup = Lines.from_mesh_geometry(mesh_geometry, dedup_edges=False)
    print(f"lines: without dedup {len(lines_without_dedup.geometry.vertices//2)} line segments")
    print(f"lines: with dedup {len(lines.geometry.vertices // 2)} line segments")
    print(f"lines reduction: {100 - (len(lines.geometry.vertices) / len(lines_without_dedup.geometry.vertices) * 100):.2f}%")

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
