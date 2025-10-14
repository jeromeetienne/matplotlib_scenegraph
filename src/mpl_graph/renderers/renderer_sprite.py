# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.image
import numpy as np

# local imports
from ..objects.sprite import Sprite
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils


class RendererSprite:
    @staticmethod
    def render(renderer: "Renderer", sprite: Sprite, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        material = sprite.material

        # =============================================================================
        # Create artists if needed
        # =============================================================================
        if sprite.uuid not in renderer._artists:
            fake_texture = np.zeros((1, 1, 3), dtype=np.uint8)
            mpl_axes_image = renderer._axis.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))
            mpl_axes_image.set_visible(False)  # hide until properly positioned and sized
            renderer._artists[sprite.uuid] = mpl_axes_image

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_axes_image = typing.cast(matplotlib.image.AxesImage, renderer._artists[sprite.uuid])
        mpl_axes_image.set_visible(True)

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        vertices = np.array([sprite.position])

        # full_transform = sprite.get_world_matrix()
        full_transform = TransformUtils.compute_mvp_matrix(camera, sprite)
        vertices = GeometryUtils.apply_transform(vertices, full_transform)

        # dispatch the post_transforming event
        sprite.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices)

        vertices_2d = vertices[:, :2]  # drop z for 2D rendering

        # =============================================================================
        # Update the artist
        # =============================================================================

        mpl_axes_image.set_array(material.texture.data)

        transformed_extent = (
            vertices_2d[0, 0] - 0.5 * sprite.scale[0] * material.texture.aspect_ratio(),
            vertices_2d[0, 0] + 0.5 * sprite.scale[0] * material.texture.aspect_ratio(),
            vertices_2d[0, 1] - 0.5 * sprite.scale[1] * 1.0,
            vertices_2d[0, 1] + 0.5 * sprite.scale[1] * 1.0,
        )
        mpl_axes_image.set_extent(transformed_extent)

        return [mpl_axes_image]
