# stdlib imports
import os
from typing import Sequence

# pip imports
import numpy as np


# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_orthographic import CameraOrthographic
from mpl_graph.renderers.renderer import Renderer
from mpl_graph.objects.polygons import Polygons
from mpl_graph.core.geometry import Geometry
from common.animation_loop import AnimationLoop
from common.scene_examples import SceneExamples
from common.mesh_parser_obj_manual import MeshParserObjManual


__dirname__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__dirname__, "../assets")
models_path = os.path.join(assets_path, "models")


def main():
    # =============================================================================
    # Setup the scene
    # =============================================================================
    scene = Object3D()

    camera = CameraOrthographic()
    scene.add_child(camera)
    camera.position[2] = 5.0

    # Create a renderer
    renderer = Renderer(100, 100)
    # Create an animation loop
    animation_loop = AnimationLoop(renderer)

    # =============================================================================
    # Add objects
    # =============================================================================

    if True:
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
        geometry = Geometry(vertices=vertices)
        quad_polygons = Polygons(geometry, polygon_count, vertices_per_polygon)
        scene.add_child(quad_polygons)

        def quad_polygons_update(delta_time: float, time_stamp: float) -> Sequence[Object3D]:
            quad_polygons.position[0] = np.cos(time_stamp) * 0.5
            return [quad_polygons]

        animation_loop.add_callback(quad_polygons_update)

    if True:
        # Load a model from an .obj file
        # file_path = os.path.join(models_path, "cube_meshio.obj")
        file_path = os.path.join(models_path, "suzanne_meshio.obj")

        polygons = SceneExamples.polygons_from_obj(file_path)
        polygons.scale[:] = 0.5
        scene.add_child(polygons)

        def polygons_update(delta_time: float, time_stamp: float) -> Sequence[Object3D]:
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
