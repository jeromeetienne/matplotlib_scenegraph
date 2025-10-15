# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.lines

from mpl_graph.renderers.renderer_utils import RendererUtils

# local imports
from ..objects.lines import Lines
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..math.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils


class RendererLines:
    @staticmethod
    def render(renderer: "Renderer", lines: Lines, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = lines.geometry
        material = lines.material
        line_count = len(geometry.vertices) // 2

        # =============================================================================
        # Sanity checks
        # =============================================================================

        assert line_count * 2 == len(geometry.vertices), "Lines vertices length must be even"

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # full_transform = lines.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, lines)
        vertices_ndc, vertices_world = GeometryUtils.apply_mvp_matrix(geometry.vertices, mvp_matrix)

        # dispatch the post_transforming event
        lines.post_transform.dispatch(vertices_ndc)

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        # drop z for 2D rendering
        vertices_2d = vertices_ndc[:, :2]

        # reshape to (line_count, 2 endpoints, 2 coords)
        vertices_2d = vertices_2d.reshape((line_count, 2, 2))

        # =============================================================================
        # Create artists if needed
        # =============================================================================

        if lines.uuid not in renderer._artists:
            mpl_line_collection = matplotlib.collections.LineCollection([])
            mpl_line_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_line_collection)
            renderer._artists[lines.uuid] = mpl_line_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_line_collection = typing.cast(matplotlib.collections.LineCollection, renderer._artists[lines.uuid])
        mpl_line_collection.set_visible(True)

        # =============================================================================
        # do z-ordering based on distance to camera
        # =============================================================================

        # compute and set zorder on our single artist
        RendererUtils.update_single_artist_zorder(camera, lines, mpl_line_collection)

        # =============================================================================
        # Update the artists
        # =============================================================================
        mpl_line_collection.set_segments(typing.cast(list, vertices_2d))
        mpl_line_collection.set_color(typing.cast(list, material.colors))
        mpl_line_collection.set_linewidth([1])  # TODO put that into material
        mpl_line_collection.set_linestyle("solid")  # TODO put that into material

        # Return the changed_artists
        return [mpl_line_collection]
