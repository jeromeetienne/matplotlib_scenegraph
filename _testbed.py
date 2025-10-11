# stdlib imports
import os

# pip imports
import numpy as np


# local imports
from core.object_3d import Object3D
from cameras.camera_orthographic import CameraOrthographic
from renderers.matplotlib.renderer import RendererMatplotlib
from helpers.animation_loop import AnimationLoop
from objects.lines import Lines
from helpers.scene_examples import SceneExamples
from objects.polygons import Polygons
from helpers.mesh_parser_obj_manual import MeshParserObjManual


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = RendererMatplotlib(512, 512)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Load a model
    # =============================================================================

    if False:
        head_points = SceneExamples.getHeadPoints()
        scene.add_child(head_points)

        def head_points_update(delta_time: float, time_stamp: float) -> list[Object3D]:
            head_points.position[0] = np.cos(time_stamp) * 0.5
            head_points.scale[:] = np.cos(time_stamp) * 0.5 + 1.5
            return [head_points]

        animation_loop.add_callback(head_points_update)

    # =============================================================================
    # Random animated lines
    # =============================================================================

    if False:
        num_lines = 10
        vertices = np.random.uniform(-1, 1, size=(num_lines * 2, 3)).astype(np.float32)

        lines = Lines(vertices)
        scene.add_child(lines)

        def lines_update(delta_time: float, time_stamp: float) -> list[Object3D]:
            lines.vertices = np.random.uniform(-1, 1, size=(num_lines * 2, 3)).astype(np.float32)
            return [lines]

        animation_loop.add_callback(lines_update)

    if False:
        # Create a list of polygons, each polygon is a list of (x,y) points
        # Add a z=0 to each (x, y) point to make (x, y, z)
        vertices = (
            np.array(
                [
                    [(1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)],
                    [(3, 1, 0), (4, 1, 0), (4, 2, 0), (3, 2, 0)],
                    [(1, 3, 0), (2, 3, 0), (2, 4, 0), (1, 4, 0)],
                ],
                dtype=np.float32,
            )
            / 5
        )

        polygon_count = vertices.shape[0]
        vertices_per_polygon = vertices.shape[1]
        vertices = vertices.reshape(polygon_count * vertices_per_polygon, 3)
        polygons = Polygons(vertices, polygon_count, vertices_per_polygon)
        scene.add_child(polygons)

        def polygons_update(delta_time: float, time_stamp: float) -> list[Object3D]:
            polygons.position[0] = np.cos(time_stamp) * 0.5
            return [polygons]

        animation_loop.add_callback(polygons_update)

    if True:
        # Load a model from an .obj file
        file_path = os.path.join(os.path.dirname(__file__), "assets", "models", "cube_meshio.obj")

        polygons = SceneExamples.polygons_from_obj(file_path)
        polygons.scale[:] = 0.5
        scene.add_child(polygons)

        def polygons_update(delta_time: float, time_stamp: float) -> list[Object3D]:
            polygons.rotation_euler[1] = time_stamp
            polygons.position[1] = np.cos(time_stamp * 3) * 1
            return [polygons]

        animation_loop.add_callback(polygons_update)

    # =============================================================================
    # Start the animation loop
    # =============================================================================
    animation_loop.start(scene, camera)


if __name__ == "__main__":
    main()
