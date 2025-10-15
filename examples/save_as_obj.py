"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
from typing import Sequence
import time

# pip imports
import os
import numpy as np
import matplotlib.pyplot

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_perspective import CameraPerspective
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects import Points, Scene
from common.example_utils import ExamplesUtils
from common.mesh_utils import MeshUtils
from common.geometry_shape import GeometryShape

__dirname__ = os.path.abspath(os.path.dirname(__file__))
output_path = os.path.join(__dirname__, "output")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================

    # Create a renderer
    renderer = Renderer(512, 512)

    # Create the scene root
    scene = Scene()

    # Create a camera
    camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # =============================================================================
    # Build your scene
    # =============================================================================

    mesh_geometry = GeometryShape.box(1, 1, 1, 10, 10, 10)
    mesh_geometry = GeometryShape.box(1, 1, 1, 10, 10, 10)

    obj_str = MeshUtils.geometry_as_obj_string(mesh_geometry)

    obj_filepath = os.path.join(output_path, "box.obj")
    with open(obj_filepath, "w") as f:
        f.write(obj_str)
    print(f"Saved box geometry as .obj to: {obj_filepath}")


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
