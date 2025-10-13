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

    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add objects
    # =============================================================================

    # Load a model from an .obj file
    # file_path = os.path.join(models_path, "cube_meshio.obj")
    file_path = os.path.join(models_path, "suzanne.obj")

    # parse the .obj file
    geometry = MeshUtils.parse_obj_file_manual(file_path)

    # Normalize the vertices to fit in a unit cube
    geometry.vertices = GeometryUtils.normalize_vertices_to_unit_cube(geometry.vertices)

    # Create a polygons object
    polygons = Polygons.from_mesh_geometry(geometry)
    # polygons.scale[:] = 0/
    scene.add_child(polygons)

    @animation_loop.decorator
    def polygons_update(delta_time: float) -> Sequence[Object3D]:
        present = time.time()
        polygons.rotation_euler[1] = present
        polygons.position[1] = np.cos(present * 3) * 1
        return [polygons]

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
