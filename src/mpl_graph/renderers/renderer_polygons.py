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
from ..geometry.geometry_utils import GeometryUtils
from ..core.constants import Constants


class RendererPolygons:
    @staticmethod
    def render(renderer: "Renderer", polygons: Polygons, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = polygons.geometry
        material = polygons.material
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
        # Apply full transform the vertices
        # =============================================================================

        # full_transform = polygons.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, polygons)
        vertices_transformed = GeometryUtils.apply_transform(geometry.vertices, mvp_matrix)
        faces_vertices = vertices_transformed.reshape(polygons.polygon_count, polygons.vertices_per_polygon, 3)  # [P, V, 3]

        # dispatch the post_transforming event
        polygons.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices_transformed)

        # =============================================================================
        # Compute face normals - needed for lighting and back-face culling
        # =============================================================================
        if material.face_culling != Constants.FaceCulling.BothSides:
            faces_normals = np.cross(
                faces_vertices[:, 2] - faces_vertices[:, 0],
                faces_vertices[:, 1] - faces_vertices[:, 0],
            )
            faces_normals_unit = faces_normals / np.linalg.norm(faces_normals, axis=1).reshape(len(faces_normals), 1)

            # =============================================================================
            # Face culling
            # =============================================================================

            # camera_cosines is the cosine of the angle between the normal and the camera
            # - if <= 0, the face is pointing away from the camera
            # - if > 0, the face is pointing towards the camera
            # camera_direction = (0, 0, -1)
            camera_direction = polygons.get_world_position() - camera.get_world_position()
            camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

            # determine which faces are hidden based on the face_culling mode
            faces_hidden = camera_cosines <= 0 if material.face_culling == Constants.FaceCulling.BackSide else camera_cosines >= 0
        else:
            # no face hidden - all False
            faces_hidden = np.zeros(shape=(len(faces_vertices),), dtype=bool)

        # log how many faces are visible
        print(f"Rendering {np.sum(~faces_hidden)} visible faces out of {polygons.polygon_count} polygons")

        # remove hidden faces
        faces_vertices = faces_vertices[~faces_hidden]

        # =============================================================================
        # Depth sort at the faces level
        # =============================================================================

        # Sort polygons by depth (painter's algorithm)
        # FIXME how does it interact with the zorder at the object3D level ?
        if material.face_sorting:
            # compute the depth of each face as the mean z value of its vertices
            faces_depth = faces_vertices[:, :, 2].mean(axis=1)
            # get the sorting indices (from farthest to nearest)
            depth_sorted_indices = np.argsort(faces_depth)
            # apply the sorting to faces_vertices and faces_hidden
            faces_vertices = faces_vertices[depth_sorted_indices]

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        vertices_2d = faces_vertices[:, :, :2]  # drop z for 2D rendering

        # =============================================================================
        # Update all the artists
        # =============================================================================

        # update the PathCollection with the new patches
        mpl_poly_collection.set_verts(typing.cast(list, vertices_2d))
        mpl_poly_collection.set_facecolor(typing.cast(list, material.colors))
        mpl_poly_collection.set_edgecolor((0, 0, 0, 0.3))
        mpl_poly_collection.set_linewidth(0.5)

        return [mpl_poly_collection]
