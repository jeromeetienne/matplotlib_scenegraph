# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.lines
import matplotlib.patches
import numpy as np

# local imports
from ..objects.polygons import Polygons
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils


class RendererPolygons:
    @staticmethod
    def render(renderer: "Renderer", polygons: Polygons, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = polygons.geometry
        # =============================================================================
        # Create artists if needed
        # =============================================================================
        if polygons.uuid not in renderer._artists:
            mpl_path_collection = matplotlib.collections.PatchCollection([])
            mpl_path_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_path_collection)
            renderer._artists[polygons.uuid] = mpl_path_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_path_collection = typing.cast(matplotlib.collections.PatchCollection, renderer._artists[polygons.uuid])
        mpl_path_collection.set_visible(True)

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # full_transform = polygons.get_world_matrix()
        full_transform = TransformUtils.compute_full_transform(camera, polygons)

        vertices_transformed = TransformUtils.apply_transform(polygons.geometry.vertices, full_transform)
        faces_vertices = vertices_transformed.reshape(polygons.polygon_count, polygons.vertices_per_polygon, 3)  # [P, V, 3]

        # dispatch the post_transforming event
        polygons.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices_transformed)

        # Sort polygons by depth (painter's algorithm)
        sort_faces = False
        if sort_faces:
            faces_depth = faces_vertices[:, :, 2].mean(axis=1)
            depth_sorted_indices = np.argsort(faces_depth)
            faces_vertices = faces_vertices[depth_sorted_indices]

        vertices_2d = faces_vertices[:, :, :2]  # drop z for 2D rendering

        # Create a polygon patch for each set of coordinates
        mpl_patches_polygons = [matplotlib.patches.Polygon(coords, closed=True) for coords in vertices_2d]

        mpl_path_collection.set_paths(mpl_patches_polygons)
        mpl_path_collection.set_facecolor(polygons.color.tolist())
        mpl_path_collection.set_facecolor((0.5, 0.5, 0.5, 1))
        mpl_path_collection.set_edgecolor((0, 0, 0, 0.3))
        mpl_path_collection.set_linewidth(0.5)

        return [mpl_path_collection]
