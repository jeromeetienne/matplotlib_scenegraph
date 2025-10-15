# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.image
import numpy as np

from mpl_graph.renderers.renderer_utils import RendererUtils

# local imports
from ..objects.sprite import Sprite
from ..renderers.renderer import Renderer
from ..cameras.camera import Camera
from ..math.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils


class RendererSprite:
    @staticmethod
    def render(renderer: "Renderer", sprite: Sprite, camera: Camera) -> list[matplotlib.artist.Artist]:
        material = sprite.material

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # Get the local position of the sprite (single vertex)
        vertices_localspace = np.array([sprite.position])

        # full_transform = sprite.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, sprite)
        vertices_ndc, vertices_clip = GeometryUtils.apply_mvp_matrix(vertices_localspace, mvp_matrix)

        # dispatch the post_transforming event
        sprite.post_transform.dispatch(vertices_clip)

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        # drop z for 2D rendering
        vertices_2d = vertices_ndc[:, :2]

        # =============================================================================
        # Compute extent
        # =============================================================================

        # compute distance from camera to object3d
        camera_position = camera.get_world_position()
        object_position = sprite.get_world_position()
        distance_to_camera = ((camera_position - object_position) ** 2).sum() ** 0.5

        extent_2d = (
            vertices_2d[0, 0] - (0.5 * sprite.scale[0] * material.extent[0] * material.texture.aspect_ratio()) / distance_to_camera,
            vertices_2d[0, 0] + (0.5 * sprite.scale[0] * material.extent[0] * material.texture.aspect_ratio()) / distance_to_camera,
            vertices_2d[0, 1] - (0.5 * sprite.scale[1] * material.extent[0] * 1.0) / distance_to_camera,
            vertices_2d[0, 1] + (0.5 * sprite.scale[1] * material.extent[0] * 1.0) / distance_to_camera,
        )

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
        # do z-ordering based on distance to camera
        # =============================================================================

        # compute and set zorder on our single artist
        RendererUtils.update_single_artist_zorder(camera, sprite, mpl_axes_image)

        # =============================================================================
        # Update the artist
        # =============================================================================

        mpl_axes_image.set_array(material.texture.data)
        mpl_axes_image.set_extent(extent_2d)

        return [mpl_axes_image]
