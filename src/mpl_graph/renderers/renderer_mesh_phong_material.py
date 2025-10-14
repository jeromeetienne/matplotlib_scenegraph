# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import numpy as np
from pyrr import vector, vector3

# local imports
from ..objects.mesh import Mesh
from ..lights import Light, DirectionalLight, PointLight, AmbientLight
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core import Object3D
from ..materials import MeshPhongMaterial
from .renderer_mesh import RendererMesh


class RendererMeshPhongMaterial:
    @staticmethod
    def shade_faces_flat(
        camera: CameraBase,
        material: MeshPhongMaterial,
        normals_world: np.ndarray,
        face_centroids_world: np.ndarray,
        lights: list[Light],
    ) -> np.ndarray:
        """
        Flat shading per face using isinstance for light types.
        Diffuse + ambient only, no specular.

        normals_world: [F, 3]  -> unit normal per face
        face_centroids_world: [F, 3] -> centroid per face
        lights: list of light objects (AmbientLight, DirectionalLight, PointLight)
        base_color: np.array([3]) RGB
        """
        num_faces = normals_world.shape[0]
        shaded = np.zeros((num_faces, 3), dtype=np.float32)

        base_color_rgb = np.array(material.color[:3], dtype=np.float32)

        # --- Ambient lights
        for light in lights:
            if isinstance(light, AmbientLight):
                light_color_rgb = np.array(light.color[:3], dtype=np.float32)
                shaded += base_color_rgb * light_color_rgb * light.intensity

        # --- Directional and Point lights
        for light in lights:
            if isinstance(light, DirectionalLight):
                # Light direction toward scene origin (or target)
                target = np.array([0, 0, 0], dtype=np.float32)
                L = target - light.get_world_position()
                L = L / np.linalg.norm(L)
                L_dir = np.tile(L, (num_faces, 1))
                attenuation = 1.0

            elif isinstance(light, PointLight):
                # Vector from face centroid to point light
                L_dir = light.get_world_position() - face_centroids_world
                dist = np.linalg.norm(L_dir, axis=1, keepdims=True) + 1e-6
                L_dir = L_dir / dist
                attenuation = 1.0 / (dist * dist)

            else:
                continue

            # --- Diffuse Lambert
            ndotl = np.clip(np.sum(normals_world * L_dir, axis=1, keepdims=True), 0, 1)
            light_color_rgb = np.array(light.color[:3], dtype=np.float32)
            diffuse = base_color_rgb * light_color_rgb * light.intensity * ndotl * attenuation
            shaded += diffuse

            # --- Specular Phong
            V = camera.get_world_position() - face_centroids_world
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
            R = 2 * ndotl * normals_world - L_dir
            R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-6)
            spec_angle = np.clip(np.sum(R * V, axis=1, keepdims=True), 0, 1)
            specular = light_color_rgb * (spec_angle**material.shininess) * attenuation
            shaded += specular

        return np.clip(shaded, 0, 1)

    # =============================================================================
    #
    # =============================================================================

    @staticmethod
    def render(
        renderer: "Renderer",
        mesh: Mesh,
        camera: CameraBase,
        faces_vertices_world: np.ndarray,
        faces_vertices_ndc: np.ndarray,
        faces_vertices_2d: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:
        material = typing.cast(MeshPhongMaterial, mesh.material)

        # =============================================================================
        # Lighting
        # =============================================================================

        # get the scene lights in the scene graph
        scene = mesh.root()
        assert isinstance(scene, Object3D)
        lights: list[Light] = [child for child in scene.traverse() if isinstance(child, Light)]

        # compute face normals and centroids in world space
        faces_normals_unit = RendererMesh.compute_faces_normal_unit(faces_vertices_world)
        faces_centroids_world = RendererMesh.compute_faces_centroids(faces_vertices_world)

        shaded_colors = RendererMeshPhongMaterial.shade_faces_flat(camera, material, faces_normals_unit, faces_centroids_world, lights)
        faces_color = shaded_colors
        # =============================================================================
        #
        # =============================================================================
        # faces_normals_unit = RendererMesh.compute_faces_normal_unit(faces_vertices_world)

        # # light_direction = light_position - mesh_position
        # light_direction = np.array((1.0, 1.0, 1.0)).astype(np.float32)
        # light_direction /= np.linalg.norm(light_direction)
        # light_cosines: np.ndarray = np.dot(faces_normals_unit, light_direction)
        # light_intensities = (light_cosines + 1) / 2

        # faces_color = np.zeros((len(faces_vertices_2d), 4))
        # faces_color[:, 0] = material.color[0] * light_intensities
        # faces_color[:, 1] = material.color[1] * light_intensities
        # faces_color[:, 2] = material.color[2] * light_intensities
        # faces_color[:, 3] = 1.0  # alpha

        # =============================================================================
        # Face sorting based on depth
        # =============================================================================

        # Sort polygons by depth (painter's algorithm)
        # FIXME how does it interact with the zorder at the object3D level ?
        if material.face_sorting:
            # compute the depth of each face as the mean z value of its vertices
            faces_depth = faces_vertices_ndc[:, :, 2].mean(axis=1)
            # get the sorting indices (from farthest to nearest)
            depth_sorted_indices = np.argsort(faces_depth)
            # apply the sorting to faces_vertices and faces_hidden
            # CAUTION: here reorder ALL arrays you use below to keep them in sync
            faces_vertices_2d = faces_vertices_2d[depth_sorted_indices]
            faces_color = faces_color[depth_sorted_indices]

        # =============================================================================
        # Face culling
        # =============================================================================

        faces_visible = RendererMesh.compute_faces_visible(faces_vertices_2d, material.face_culling)

        # keep only visible faces
        # - CAUTION: here reorder ALL arrays you use below to keep them in sync
        faces_vertices_2d = faces_vertices_2d[faces_visible]
        faces_color = faces_color[faces_visible]

        # =============================================================================
        #
        # =============================================================================

        # =============================================================================
        #
        # =============================================================================

        # =============================================================================
        # Create artists if needed
        # =============================================================================
        if mesh.uuid not in renderer._artists:
            mpl_poly_collection = matplotlib.collections.PolyCollection([], clip_on=False, snap=False)
            mpl_poly_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_poly_collection)
            renderer._artists[mesh.uuid] = mpl_poly_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_poly_collection = typing.cast(matplotlib.collections.PolyCollection, renderer._artists[mesh.uuid])
        mpl_poly_collection.set_visible(True)

        # =============================================================================
        # Update all the artists
        # =============================================================================

        # update the PathCollection with the new patches
        mpl_poly_collection.set_verts(typing.cast(list, faces_vertices_2d))
        mpl_poly_collection.set_facecolor(typing.cast(list, faces_color))
        # TODO handle that and put them in the material
        mpl_poly_collection.set_edgecolor((0, 0, 0, 0.3))
        mpl_poly_collection.set_linewidth(0.5)

        return [mpl_poly_collection]
