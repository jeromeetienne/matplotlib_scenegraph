# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.lines

# local imports
from ..objects.lines import Lines
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils


class RendererLines:
    @staticmethod
    def render(renderer: "Renderer", lines: Lines, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = lines.geometry

        line_count = len(geometry.vertices) // 2
        assert line_count * 2 == len(geometry.vertices), "Lines vertices length must be even"

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
        # Apply full transform the vertices
        # =============================================================================

        # full_transform = lines.get_world_matrix()
        full_transform = TransformUtils.compute_full_transform(camera, lines)
        vertices = TransformUtils.apply_transform(geometry.vertices, full_transform)

        # dispatch the post_transforming event
        lines.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices)

        vertices_2d = vertices[:, :2]  # drop z for 2D rendering

        vertices_2d = vertices_2d.reshape((line_count, 2, 2))  # reshape to (line_count, 2 endpoints, 2 coords)
        segments = vertices_2d.tolist()
        mpl_line_collection.set_segments(segments)
        mpl_line_collection.set_color(lines.color.tolist())

        return [mpl_line_collection]
