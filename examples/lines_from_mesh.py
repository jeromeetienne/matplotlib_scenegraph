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
from common.example_utils import ExamplesUtils
from common.animation_loop import AnimationLoop
from common.mesh_utils import MeshUtils
from mpl_graph.geometry import GeometryUtils


__dirname__ = os.path.dirname(os.path.abspath(__file__))
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
    renderer = Renderer(100, 100)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Random animated lines
    # =============================================================================

    # Load a model from an .obj file
    # file_path = os.path.join(models_path, "cube_meshio.obj")
    file_path = os.path.join(models_path, "suzanne_meshio.obj")

    # parse the .obj file
    mesh_geometry = MeshUtils.parse_obj_file_manual(file_path)
    assert mesh_geometry.indices is not None, "The .obj file must contain face indices"

    print(f"mesh_geometry vertices: {mesh_geometry.vertices.shape}, faces: {mesh_geometry.indices.shape}")

    # Normalize the vertices to fit in a unit cube
    mesh_geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(mesh_geometry.vertices)

    # Build the lines object
    material = LinesMaterial(colors=np.array([Constants.Color.GRAY]))
    lines = Lines.from_mesh_geometry(mesh_geometry, dedup_edges=True)

    print(f"lines vertices: {lines.geometry.vertices.shape}")
    lines.material.colors = Constants.Color.CYAN
    lines.scale[:] = 0.5
    scene.add_child(lines)

    @animation_loop.decorator
    def lines_update(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        lines.rotation_euler[1] = present
        lines.position[1] = np.cos(present * 3) * 1
        return [lines]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
