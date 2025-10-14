# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections

from mpl_graph.geometry.geometry_utils import GeometryUtils

# local imports
from ..objects.points import Points
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils


class RendererPoints:
    @staticmethod
    def render(renderer: "Renderer", points: Points, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = points.geometry
        material = points.material

        # =============================================================================
        # Create the artists if needed
        # =============================================================================
        if points.uuid not in renderer._artists:
            mpl_path_collection = renderer._axis.scatter([], [])  # type: ignore
            mpl_path_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_path_collection)
            renderer._artists[points.uuid] = mpl_path_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================
        # Get the existing mpl artist
        mpl_path_collection = typing.cast(matplotlib.collections.PathCollection, renderer._artists[points.uuid])
        # set it to visible
        mpl_path_collection.set_visible(True)

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # full_transform = points.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, points)
        vertices_transformed = GeometryUtils.apply_transform(geometry.vertices, mvp_matrix)

        # dispatch the post_transforming event
        points.post_transform.dispatch(vertices_transformed)

        # drop z for 2D rendering
        vertices_2d = vertices_transformed[:, :2]

        # =============================================================================
        # Update the matplotlib artists data
        # =============================================================================

        mpl_path_collection.set_offsets(offsets=vertices_2d)
        mpl_path_collection.set_sizes(typing.cast(list, material.sizes))  # set a default size for each point
        mpl_path_collection.set_color(typing.cast(list, material.colors))
        mpl_path_collection.set_edgecolor(typing.cast(list, material.edge_colors))
        mpl_path_collection.set_linewidth(typing.cast(list, material.edge_widths))

        return [mpl_path_collection]
