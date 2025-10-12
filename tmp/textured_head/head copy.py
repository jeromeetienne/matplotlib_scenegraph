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


def warp_coordinates(face_coord_1: np.ndarray, face_coord_2: np.ndarray) -> matplotlib.transforms.Affine2D:
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


def textured_triangle(
    mpl_axes: matplotlib.axes.Axes,
    face_vertices: np.ndarray,
    uvs_normalized: np.ndarray,
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
    uvs_pixel = uvs_normalized * (image_w, image_h)

    x_min = int(np.floor(uvs_pixel[:, 0].min()))
    x_max = int(np.ceil(uvs_pixel[:, 0].max()))
    y_min = int(np.floor(uvs_pixel[:, 1].min()))
    y_max = int(np.ceil(uvs_pixel[:, 1].max()))

    texture = (texture[y_min:y_max, x_min:x_max, :] * intensity).astype(np.uint8)
    extent = x_min / image_w, x_max / image_w, y_min / image_h, y_max / image_h

    transform = warp_coordinates(uvs_normalized, face_vertices) + mpl_axes.transData

    path = matplotlib.path.Path(
        [uvs_normalized[0], uvs_normalized[1], uvs_normalized[2], uvs_normalized[0]],
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

vertices_coords, uvs_coords, normals_coords, faces_vertice_indices, face_uv_indices, normal_indices = obj_read(model_path)
texture = imageio.imread(texture_path)[::-1, ::1, :3]
face_vertices = vertices_coords[faces_vertice_indices]
face_uvs = uvs_coords[face_uv_indices]

# =============================================================================
# Lighting
# =============================================================================
# N is the normal to the triangle
faces_normal = np.cross(
    face_vertices[:, 2] - face_vertices[:, 0],
    face_vertices[:, 1] - face_vertices[:, 0],
)
faces_normal_unit = faces_normal / np.linalg.norm(faces_normal, axis=1).reshape(len(faces_normal), 1)
# L is the cosine of the angle between the normal and the light
light_direction = (0, 0, -1)
light_cosines = np.dot(faces_normal_unit, light_direction)

# =============================================================================
# Sort triangles by depth (painter's algorithm)
# =============================================================================
depth_sorted_indices = np.argsort(face_vertices[:, :, 2].mean(axis=1))
face_vertices_2d = face_vertices[depth_sorted_indices][..., :2]
face_uvs = face_uvs[depth_sorted_indices][..., :2]
light_cosines = light_cosines[depth_sorted_indices]

# =============================================================================
# Loop over faces and draw them
# =============================================================================
for vertices, face_uvs, light_cosine in zip(face_vertices, face_uvs, light_cosines):
    if light_cosine > 0:
        try:
            textured_triangle(mpl_axes=mpl_axes, face_vertices=vertices, uvs_normalized=face_uvs, texture=texture, intensity=(light_cosine + 1) / 2)
        except np.linalg.LinAlgError:
            pass

# plt.savefig("head.pdf")
# plt.savefig("head.png")
# plt.savefig("head.svg")
matplotlib.pyplot.show()
