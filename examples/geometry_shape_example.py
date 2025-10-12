"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
from typing import Sequence

# pip imports
import matplotlib.pyplot

# local imports
from common.geometry_shape import GeometryShape
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.textured_mesh import TexturedMesh
from mpl_graph.core.texture import Texture
from mpl_graph.core.geometry import Geometry
from common.mesh_utils import MeshUtils
from common.animation_loop import AnimationLoop

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
    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(256, 256)

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a texture
    # =============================================================================

    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path)
    # remove the alpha channel if any
    texture = texture.strip_alpha() if texture.has_alpha() else texture

    # =============================================================================
    # Add a plane with the texture
    # =============================================================================

    geometry_plane = GeometryShape.plane(1.0, 1.0)
    mesh_plane = TexturedMesh(geometry_plane, texture)
    mesh_plane.position[0] = -0.5
    mesh_plane.scale[:] = 0.2
    scene.add_child(mesh_plane)

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
