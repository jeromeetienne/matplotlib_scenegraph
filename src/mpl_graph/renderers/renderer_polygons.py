# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.lines
import matplotlib.patches
import numpy as np

from mpl_graph.renderers.renderer_utils import RendererUtils

# local imports
from ..objects.polygons import Polygons
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..math.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils
from ..core.constants import Constants


class RendererPolygons:
    @staticmethod
    def render(renderer: "Renderer", polygons: Polygons, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = polygons.geometry
        material = polygons.material

        # TODO factorize with RendererMesh

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # Get the vertices in local space
        vertices_localspace = geometry.vertices

        # TODO bug all the face culling + lighting must be done in world space

        # full_transform = polygons.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, polygons)
        vertices_ndc, vertices_clip = GeometryUtils.apply_mvp_matrix(vertices_localspace, mvp_matrix)

        # reshape to faces - shape [P, V, 3]
        faces_vertices_ndc = vertices_ndc.reshape(polygons.polygon_count, polygons.vertices_per_polygon, 3)

        # dispatch the post_transforming event
        polygons.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices_clip)

        # =============================================================================
        # Face culling
        # =============================================================================
        if material.face_culling != Constants.FaceCulling.BothSides:
            # compute the face normals in view space
            faces_normals = np.cross(
                faces_vertices_ndc[:, 2] - faces_vertices_ndc[:, 0],
                faces_vertices_ndc[:, 1] - faces_vertices_ndc[:, 0],
            )
            faces_normals_unit = faces_normals / np.linalg.norm(faces_normals, axis=1).reshape(len(faces_normals), 1)

            # =============================================================================
            # Face culling
            # - camera_cosines is the cosine of the angle between the normal and the camera
            # - if <= 0, the face is pointing away from the camera
            # - if > 0, the face is pointing towards the camera
            # camera_direction = (0, 0, -1)
            camera_direction = polygons.get_world_position() - camera.get_world_position()
            camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

            # determine which faces are hidden based on the face_culling mode
            faces_visible = camera_cosines >= 0 if material.face_culling == Constants.FaceCulling.BackSide else camera_cosines < 0
        else:
            # no face hidden - all False
            faces_visible = np.ones(shape=(len(faces_vertices_ndc),), dtype=bool)

        # log how many faces are visible
        # print(f"Rendering {np.sum(faces_visible)} visible faces out of {polygons.polygon_count} polygons")

        # remove hidden faces
        faces_vertices_ndc = faces_vertices_ndc[faces_visible]

        # =============================================================================
        # Depth sort at the faces level
        # =============================================================================

        # Sort polygons by depth (painter's algorithm)
        if material.depth_sorting:
            # compute the depth of each face as the mean z value of its vertices
            faces_depth = faces_vertices_ndc[:, :, 2].mean(axis=1)
            # get the sorting indices (from farthest to nearest)
            depth_sorted_indices = np.argsort(faces_depth)
            # apply the sorting to faces_vertices
            faces_vertices_ndc = faces_vertices_ndc[depth_sorted_indices]

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        faces_vertices_2d = faces_vertices_ndc[:, :, :2]  # drop z for 2D rendering

        # =============================================================================
        # Create artists if needed
        # =============================================================================
        if polygons.uuid not in renderer._artists:
            mpl_poly_collection = matplotlib.collections.PolyCollection([], clip_on=False, snap=False)
            mpl_poly_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_poly_collection)
            renderer._artists[polygons.uuid] = mpl_poly_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_poly_collection = typing.cast(matplotlib.collections.PolyCollection, renderer._artists[polygons.uuid])
        mpl_poly_collection.set_visible(True)

        # =============================================================================
        # do z-ordering based on distance to camera
        # =============================================================================

        # compute and set zorder on our single artist
        RendererUtils.update_single_artist_zorder(camera, polygons, mpl_poly_collection)

        # =============================================================================
        # Update all the artists
        # =============================================================================

        # update the PathCollection with the new patches
        mpl_poly_collection.set_verts(typing.cast(list, faces_vertices_2d))
        mpl_poly_collection.set_facecolor(typing.cast(list, material.colors))
        mpl_poly_collection.set_edgecolor((0, 0, 0, 0.3))
        mpl_poly_collection.set_linewidth(0.5)

        return [mpl_poly_collection]
