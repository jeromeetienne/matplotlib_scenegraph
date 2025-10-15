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


class RendererText:
    @staticmethod
    def render(renderer: "Renderer", text: Text, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        material: TextMaterial = text.material

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
        # Apply full transform the vertices
        # =============================================================================

        vertices = np.array([text.position])

        # full_transform = sprite.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, text)
        vertices = GeometryUtils.apply_transform(vertices, mvp_matrix)

        # dispatch the post_transforming event
        text.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices)

        # drop z for 2D rendering
        vertices_2d = vertices[:, :2]

        # =============================================================================
        # Update the artist
        # =============================================================================

        mpl_text.set_position((vertices_2d[0, 0], vertices_2d[0, 1]))
        mpl_text.set_text(text.content)
        mpl_text.set_fontsize(material.font_size)
        mpl_text.set_color(material.color.tolist())
        mpl_text.set_horizontalalignment(material.horizontal_align)
        mpl_text.set_verticalalignment(material.vertical_align)

        return [mpl_text]
