# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.text
import numpy as np

# local imports
from ..objects.text import Text
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..math.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils
from ..materials.text_material import TextMaterial
from .renderer_utils import RendererUtils


class RendererText:
    @staticmethod
    def render(renderer: "Renderer", text: Text, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        material: TextMaterial = text.material
        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # Get the local position of the sprite (single vertex)
        vertices_localspace = np.array([text.position])

        # full_transform = sprite.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, text)
        vertices_ndc, vertices_clip = GeometryUtils.apply_mvp_matrix(vertices_localspace, mvp_matrix)

        # dispatch the post_transforming event
        text.post_transform.dispatch(vertices_clip)

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        # drop z for 2D rendering
        vertices_2d = vertices_ndc[:, :2]

        # =============================================================================
        # Create artists if needed
        # =============================================================================
        if text.uuid not in renderer._artists:
            mpl_text = renderer._axis.text(0, 0, "")
            mpl_text.set_visible(False)  # hide until properly positioned and sized
            renderer._artists[text.uuid] = mpl_text

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_text = typing.cast(matplotlib.text.Text, renderer._artists[text.uuid])
        mpl_text.set_visible(True)

        # =============================================================================
        # do z-ordering based on distance to camera
        # =============================================================================

        # compute and set zorder on our single artist
        RendererUtils.update_single_artist_zorder(camera, text, mpl_text)

        # =============================================================================
        # Update the artists
        # =============================================================================

        mpl_text.set_position((vertices_2d[0, 0], vertices_2d[0, 1]))
        mpl_text.set_text(text.content)
        mpl_text.set_fontsize(material.font_size)
        mpl_text.set_color(material.color.tolist())
        mpl_text.set_horizontalalignment(material.horizontal_align)
        mpl_text.set_verticalalignment(material.vertical_align)

        return [mpl_text]
