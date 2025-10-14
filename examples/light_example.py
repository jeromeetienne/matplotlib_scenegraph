"""
basic example of rendering a rotating point cloud
"""

# stdlib imports
import os
import time
from typing import Sequence

# pip imports
import numpy as np

# local imports
from mpl_graph.core import Object3D, Texture
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.core.constants import Constants
from mpl_graph.renderers import Renderer
from mpl_graph.objects import Mesh, Scene
from mpl_graph.lights import DirectionalLight, PointLight, AmbientLight, Light
from mpl_graph.geometry import Geometry
from mpl_graph.materials import MeshPhongMaterial, MeshBasicMaterial, MeshNormalMaterial, MeshDepthMaterial, MeshTexturedMaterial
from common.mesh_utils import MeshUtils
from common.animation_loop import AnimationLoop
from common.example_utils import ExamplesUtils
from common.geometry_shape import GeometryShape

__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")
images_path = os.path.join(assets_path, "images")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================

    # Create a renderer
    renderer = Renderer(512, 512, background_color=Constants.Color.LIGHT_GRAY)

    # Create the scene root
    scene = Scene()

    # Create a camera and add it to the scene
    camera = CameraOrthographic()
    camera.near = 0.1
    camera.far = 4
    # camera = CameraPerspective()
    scene.add_child(camera)
    camera.position[2] = 5.0
    camera.position[1] = 2.0
    camera.rotation_euler[0] = np.pi / 10

    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    # Create a directional key light
    directional_light_key = DirectionalLight(intensity=1)
    directional_light_key.position = np.array((1.0, 0.0, -1.0))
    scene.add_child(directional_light_key)
    scene.add_child(directional_light_key.target)

    # # Create a directional fill light
    # directional_light_fill = DirectionalLight(color=Constants.Color.WHITE, intensity=0.3)
    # directional_light_fill.position = np.array((-1.0, 0.0, -1.0))
    # scene.add_child(directional_light_fill)
    # scene.add_child(directional_light_fill.target)

    # # Create a directional rim light
    # directional_light_rim = DirectionalLight(color=Constants.Color.WHITE, intensity=0.2)
    # directional_light_rim.position = np.array((0.0, 1.0, -1.0))
    # scene.add_child(directional_light_rim)
    # scene.add_child(directional_light_rim.target)

    # # add a ambient light
    # ambient_light = AmbientLight(color=Constants.Color.DARK_RED, intensity=0.2)
    # scene.add_child(ambient_light)

    # # # add a point light
    # point_light = PointLight(color=Constants.Color.RED, intensity=2)
    # point_light.position = np.array((2.0, 0.0, 2.0))
    # scene.add_child(point_light)

    # point_light_helper = Mesh(Geometry.box(), MeshBasicMaterial(color=Constants.Color.RED))

    @animation_loop.decorator
    def light_update(delta_time: float) -> Sequence[Light]:
        present = time.time()
        directional_light_key.position[0] = 2.0 * np.cos(present)
        directional_light_key.position[2] = 2.0 * np.sin(present)
        return [directional_light_key]

    mesh_grid = Mesh(GeometryShape.grid(5.0, 5.0), MeshPhongMaterial(shininess=30, color=Constants.Color.LIGHT_BLUE))
    scene.add_child(mesh_grid)
    mesh_grid.rotation_euler[1] = np.pi
    mesh_grid.position[1] = 0
    mesh_grid.position[2] = -0.01  # trick to ensure the grid is behind the other objects

    @animation_loop.decorator
    def grid_update(delta_time: float) -> Sequence[Object3D]:
        # mesh_grid.rotation_euler[0] += 0.1 * delta_time
        return [mesh_grid]

    # =============================================================================
    #
    # =============================================================================
    # Load a texture image
    texture_path = os.path.join(images_path, "uv-grid.png")
    texture = Texture.from_file(texture_path)
    # remove the alpha channel if any
    texture = texture.strip_alpha() if texture.has_alpha() else texture

    # Load a obj model
    obj_path = os.path.join(models_path, "head.obj")
    # obj_path = os.path.join(models_path, "suzanne.obj")
    # obj_path = os.path.join(models_path, "cube_meshio.obj")
    mesh_geometry = MeshUtils.parse_obj_file_manual(obj_path)

    # mesh_geometry = GeometryShape.box(1, 1, 1, 3, 3, 3)

    # Create a textured mesh
    material = MeshTexturedMaterial(texture)
    # material = MeshPhongMaterial()
    # material = MeshPhongMaterial(shininess=30, color=Constants.Color.LIGHT_BLUE)
    # material = MeshBasicMaterial()
    # material = MeshNormalMaterial()
    # material = MeshDepthMaterial(colormap_name="viridis")
    mesh_head = Mesh(mesh_geometry, material)
    mesh_head.scale[:] = 1
    mesh_head.rotation_euler[1] = np.pi

    # Add the textured mesh to the scene
    scene.add_child(mesh_head)

    @animation_loop.decorator
    def mesh_update(delta_time: float) -> list[Mesh]:
        present = time.time()
        # mesh.position[0] = np.sin(present)
        # mesh.rotation_euler[1] += delta_time
        return [mesh_head]

    # =============================================================================
    # Start the animation loop
    # =============================================================================

    # renderer.render(scene, camera)
    # print("scene rendered")
    # matplotlib.pyplot.show(block=True)

    animation_loop.start(scene, camera)


if __name__ == "__main__":
    ExamplesUtils.preamble()
    main()
