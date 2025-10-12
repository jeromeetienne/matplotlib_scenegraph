# -----------------------------------------------------------------------------
# Graphic Server Protocol (GSP)
# Copyright 2023 Vispy Development Team - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
# import matplotlib
# matplotlib.use("module://mplcairo.macosx")

# stdlib imports
import os


# pip imports
import numpy as np
import matplotlib.transforms
import matplotlib.animation
import matplotlib.image
import matplotlib.axes
import matplotlib.path
import matplotlib.pyplot

__dirname__ = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# OBJ file reader
# =============================================================================
def parse_model_obj(filename):
    """
    Read a wavefront filename and returns vertices, texcoords and
    respective indices for faces and texcoords
    """

    vertices_coords, uvs_coords, normals_coords, faces_vertice_indices, face_uv_indices, face_normal_indices = [], [], [], [], [], []
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                vertices_coords.append([float(x) for x in values[1:4]])
            elif values[0] == "vt":
                uvs_coords.append([float(x) for x in values[1:3]])
            elif values[0] == "vn":
                normals_coords.append([float(x) for x in values[1:4]])
            elif values[0] == "f":
                faces_vertice_indices.append([int(indices.split("/")[0]) for indices in values[1:]])
                face_uv_indices.append([int(indices.split("/")[1]) for indices in values[1:]])
                face_normal_indices.append([int(indices.split("/")[2]) for indices in values[1:]])
    return (
        np.array(vertices_coords),
        np.array(uvs_coords),
        np.array(normals_coords),
        np.array(faces_vertice_indices) - 1,
        np.array(face_uv_indices) - 1,
        np.array(face_normal_indices) - 1,
    )


# =============================================================================
# Affine transform to warp triangles
# =============================================================================
def texture_coords_wrap(face_coord_1: np.ndarray, face_coord_2: np.ndarray) -> matplotlib.transforms.Affine2D | None:
    """
    Return an affine transform that warp triangle T1 into triangle
    T2.

    return None if `LinAlgError` if T1 or T2 are degenerated triangles
    """

    face_coord_1 = np.c_[np.array(face_coord_1), np.ones(3)]
    face_coord_2 = np.c_[np.array(face_coord_2), np.ones(3)]
    try:
        matrix = np.linalg.inv(face_coord_1) @ face_coord_2
    except np.linalg.LinAlgError:
        return None

    return matplotlib.transforms.Affine2D(matrix.T)


# =============================================================================
#
# =============================================================================


def create_textured_triangle(mpl_axes: matplotlib.axes.Axes) -> matplotlib.image.AxesImage:
    fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
    axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))
    return axes_image


def update_textured_triangle(
    mpl_axes: matplotlib.axes.Axes,
    axes_image: matplotlib.image.AxesImage,
    face_vertices: np.ndarray,
    face_uvs: np.ndarray,
    texture: np.ndarray,
    intensity: np.float64,
    interpolation="none",
) -> None:
    """
    Parameters
    ----------
    T : (3,2) np.ndarray
      Positions of the triangle vertices
    UV : (3,2) np.ndarray
      UV coordinates of the triangle vertices
    texture:
      Image to use for texture
    """

    image_w, image_h = texture.shape[:2]
    uvs_pixel = face_uvs * (image_w, image_h)

    x_min = int(np.floor(uvs_pixel[:, 0].min()))
    x_max = int(np.ceil(uvs_pixel[:, 0].max()))
    y_min = int(np.floor(uvs_pixel[:, 1].min()))
    y_max = int(np.ceil(uvs_pixel[:, 1].max()))

    texture = (texture[y_min:y_max, x_min:x_max, :] * intensity).astype(np.uint8)
    extent = x_min / image_w, x_max / image_w, y_min / image_h, y_max / image_h

    # fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
    # axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))

    matrix_wrap = texture_coords_wrap(face_uvs, face_vertices)
    if matrix_wrap is None:
        # if degenerated triangle, hide the image
        axes_image.set_extent((0, 0, 0, 0))
        return

    transform = matrix_wrap + mpl_axes.transData

    path = matplotlib.path.Path(
        [face_uvs[0], face_uvs[1], face_uvs[2], face_uvs[0]],
        closed=True,
    )

    axes_image.set_data(texture)
    axes_image.set_interpolation(interpolation)
    axes_image.set_extent(extent)
    axes_image.set_transform(transform)
    axes_image.set_clip_path(path, transform)


class Mesh:
    def __init__(self, faces_vertices: np.ndarray, faces_uvs: np.ndarray, texture: np.ndarray):
        self.faces_vertices = faces_vertices
        self.faces_uvs = faces_uvs
        self.texture = texture

        assert faces_vertices.ndim == 3 and faces_vertices.shape[1:] == (3, 3)
        assert faces_uvs.ndim == 3 and faces_uvs.shape[1:] == (3, 2)
        assert texture.ndim == 3 and texture.shape[2] == 3

        assert len(faces_vertices) == len(faces_uvs)

    @staticmethod
    def from_obj(model_path: str, texture_path: str) -> "Mesh":
        vertices_coords, uvs_coords, normals_coords, faces_vertice_indices, face_uv_indices, face_normal_indices = parse_model_obj(model_path)
        faces_vertices = vertices_coords[faces_vertice_indices]
        faces_uvs = uvs_coords[face_uv_indices]

        texture = matplotlib.pyplot.imread(texture_path) * 255
        # remove the alpha channel if any
        texture = texture[::-1, ::1, :3]

        return Mesh(faces_vertices=faces_vertices, faces_uvs=faces_uvs, texture=texture)


class MeshRenderer:
    def __init__(self, mesh: Mesh, mpl_axes: matplotlib.axes.Axes):
        self.mesh = mesh
        self.mpl_axes = mpl_axes

        # Create a list of axes images for each face
        self.axes_images: list[matplotlib.image.AxesImage] = []
        faces_count = len(mesh.faces_vertices)
        for _ in range(faces_count):
            axes_image = create_textured_triangle(mpl_axes)
            self.axes_images.append(axes_image)

    def update(self, mesh_position: np.ndarray, camera_position: np.ndarray, light_position: np.ndarray) -> list[matplotlib.image.AxesImage]:

        assert mesh_position.shape == (3,)
        assert camera_position.shape == (3,)
        assert light_position.shape == (3,)

        mesh = self.mesh
        mpl_axes = self.mpl_axes

        faces_vertices = mesh.faces_vertices.copy()
        faces_uvs = mesh.faces_uvs.copy()
        texture = mesh.texture

        # =============================================================================
        # Compute face normals - needed for lighting and back-face culling
        # =============================================================================
        faces_normals = np.cross(
            faces_vertices[:, 2] - faces_vertices[:, 0],
            faces_vertices[:, 1] - faces_vertices[:, 0],
        )
        faces_normals_unit = faces_normals / np.linalg.norm(faces_normals, axis=1).reshape(len(faces_normals), 1)

        # =============================================================================
        # Face culling
        # =============================================================================

        # camera_cosines is the cosine of the angle between the normal and the camera
        # camera_direction = (0, 0, -1)
        camera_direction = camera_position - mesh_position
        camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

        faces_hidden = camera_cosines <= 0

        # back face culling
        # faces_vertices = faces_vertices[camera_cosines > 0]
        # faces_uvs = faces_uvs[camera_cosines > 0]
        # faces_normals_unit = faces_normals_unit[camera_cosines > 0]

        # =============================================================================
        # Lighting
        # =============================================================================
        light_direction = light_position - mesh_position
        light_direction_unit = light_direction / np.linalg.norm(light_direction)
        light_cosines: np.ndarray = np.dot(faces_normals_unit, light_direction_unit)
        light_intensities = (light_cosines + 1) / 2

        # =============================================================================
        # Sort triangles by depth (painter's algorithm)
        # =============================================================================
        faces_depth = faces_vertices[:, :, 2].mean(axis=1)
        depth_sorted_indices = np.argsort(faces_depth)
        faces_vertices = faces_vertices[depth_sorted_indices][..., :2]
        faces_uvs = faces_uvs[depth_sorted_indices][..., :2]
        light_intensities = light_intensities[depth_sorted_indices]
        faces_hidden = faces_hidden[depth_sorted_indices]

        # =============================================================================
        # Loop over faces and draw them
        # =============================================================================
        for face_index, (face_vertices, face_uvs, light_intensity, face_hidden) in enumerate(zip(faces_vertices, faces_uvs, light_intensities, faces_hidden)):
            if face_hidden:
                continue
            axes_image = self.axes_images[face_index]
            update_textured_triangle(
                mpl_axes=mpl_axes,
                axes_image=axes_image,
                face_vertices=face_vertices,
                face_uvs=face_uvs,
                texture=texture,
                intensity=light_intensity,
            )
        return self.axes_images


# =============================================================================
# main function
# =============================================================================
def main():

    # =============================================================================
    # init matplotlib figure
    # =============================================================================
    mpl_figure = matplotlib.pyplot.figure(figsize=(3, 3), dpi=100)
    mpl_axes = mpl_figure.add_axes((0, 0, 1, 1))
    mpl_axes.set_xlim(-1, 1)
    mpl_axes.set_xticks([])
    mpl_axes.set_ylim(-1, 1)
    mpl_axes.set_yticks([])

    # =============================================================================
    # Load model
    # =============================================================================
    model_path = os.path.join(__dirname__, "head.obj")
    texture_path = os.path.join(__dirname__, "uv-grid.png")

    mesh = Mesh.from_obj(model_path, texture_path)

    mesh_renderer = MeshRenderer(mesh, mpl_axes)
    mesh_position = np.array([0, 0, 0]).astype(np.float64)
    camera_position = np.array([0, 0, -2]).astype(np.float64)
    light_position = np.array([1, 0, 0]).astype(np.float64)

    # define a matplotlib animation function
    # angle = 0.0
    # angle = angle + np.pi / 2
    # camera_position[0] = 2 * np.sin(angle)
    # camera_position[1] = 2 * np.cos(angle)

    angle_rad = 0.0

    last_render_time = None
    import time

    def mpl_update(frame):
        nonlocal angle_rad, last_render_time
        print(f"Frame {frame}")

        delta_time = (time.perf_counter() - last_render_time) if last_render_time is not None else 1 / 60
        last_render_time = time.perf_counter()

        angle_rad += (2 * np.pi) / (5 / delta_time)  # one turn every 5 seconds

        light_position[0] = 2 * np.sin(angle_rad)
        light_position[2] = 2 * np.cos(angle_rad)

        axes_images = mesh_renderer.update(mesh_position, camera_position, light_position)
        return axes_images

    animation = matplotlib.animation.FuncAnimation(mpl_figure, mpl_update, frames=100, interval=1)

    matplotlib.pyplot.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    main()
