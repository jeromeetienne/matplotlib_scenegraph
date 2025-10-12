# -----------------------------------------------------------------------------
# Graphic Server Protocol (GSP)
# Copyright 2023 Vispy Development Team - BSD 2 Clauses licence
# -----------------------------------------------------------------------------
# import matplotlib
# matplotlib.use("module://mplcairo.macosx")

import types
import numpy as np


import os

__dirname__ = os.path.dirname(os.path.abspath(__file__))


import imageio.v3 as imageio
import matplotlib.transforms
import matplotlib.axes
import matplotlib.path
import matplotlib.pyplot

# from matplotlib.backend_bases import GraphicsContextBase, RendererBase


# class GC(GraphicsContextBase):
#     def __init__(self):
#         super().__init__()
#         self._antialias = False

# def custom_new_gc(self):
#     return GC()

# RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)


# =============================================================================
# OBJ file reader
# =============================================================================
def obj_read(filename):
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
def texture_coords_wrap(face_coord_1: np.ndarray, face_coord_2: np.ndarray) -> matplotlib.transforms.Affine2D:
    """
    Return an affine transform that warp triangle T1 into triangle
    T2.

    Raises
    ------

    `LinAlgError` if T1 or T2 are degenerated triangles
    """

    face_coord_1 = np.c_[np.array(face_coord_1), np.ones(3)]
    face_coord_2 = np.c_[np.array(face_coord_2), np.ones(3)]
    matrix = np.linalg.inv(face_coord_1) @ face_coord_2
    return matplotlib.transforms.Affine2D(matrix.T)


# =============================================================================
# textured triangle function
# =============================================================================
def render_textured_triangle(
    mpl_axes: matplotlib.axes.Axes,
    face_vertices: np.ndarray,
    face_uvs: np.ndarray,
    texture: np.ndarray,
    intensity: np.float64,
    interpolation="none",
):
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

    transform = texture_coords_wrap(face_uvs, face_vertices) + mpl_axes.transData

    path = matplotlib.path.Path(
        [face_uvs[0], face_uvs[1], face_uvs[2], face_uvs[0]],
        closed=True,
    )

    axes_image = mpl_axes.imshow(
        texture,
        interpolation=interpolation,
        origin="lower",
        extent=extent,
        transform=transform,
        clip_path=(path, transform),
    )


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

vertices_coords, uvs_coords, normals_coords, faces_vertice_indices, face_uv_indices, face_normal_indices = obj_read(model_path)
texture = imageio.imread(texture_path)[::-1, ::1, :3]
faces_vertices = vertices_coords[faces_vertice_indices]
faces_uvs = uvs_coords[face_uv_indices]

# =============================================================================
# Compute face normals
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
camera_direction = (0, 0, -1)
camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

# back face culling
faces_vertices = faces_vertices[camera_cosines > 0]
faces_uvs = faces_uvs[camera_cosines > 0]

# =============================================================================
# Lighting
# =============================================================================
light_direction = np.array([1, 1, -1])
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

# =============================================================================
# Loop over faces and draw them
# =============================================================================
for face_vertices, face_uvs, light_intensity in zip(faces_vertices, faces_uvs, light_intensities):
    try:
        render_textured_triangle(mpl_axes=mpl_axes, face_vertices=face_vertices, face_uvs=face_uvs, texture=texture, intensity=light_intensity)
    except np.linalg.LinAlgError:
        pass

# =============================================================================
# Save or show figure
# =============================================================================
# plt.savefig("head.pdf")
# plt.savefig("head.png")
# plt.savefig("head.svg")
matplotlib.pyplot.show()
