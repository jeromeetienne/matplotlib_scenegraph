# scene_graph_pyrr_matrix44.py
import numpy as np
import matplotlib.pyplot as plt
from pyrr import matrix44


# ---------------------------
# Utilities
# ---------------------------
def vec3(x=0.0, y=0.0, z=0.0):
    """Return a 3-element numpy vector (float64)."""
    return np.array([x, y, z], dtype=float)


def to_homogeneous(points3):
    """points3: (N,3) -> (N,4) homogeneous coordinates"""
    n = points3.shape[0]
    return np.hstack([points3, np.ones((n, 1), dtype=float)])


# ---------------------------
# Scene graph core
# ---------------------------
class Object3D:
    def __init__(self, name="Object3D"):
        self.name = name
        # plain numpy arrays for transforms
        self.position = vec3(0.0, 0.0, 0.0)
        self.rotation_euler = vec3(0.0, 0.0, 0.0)  # radians, order: (x, y, z)
        self.scale = vec3(1.0, 1.0, 1.0)

        self.parent = None
        self.children = []

    def add(self, child: "Object3D"):
        child.parent = self
        self.children.append(child)

    # ---------- matrix building using pyrr.matrix44 ----------
    def get_local_matrix(self):
        """Return a 4x4 local transform matrix (numpy array)"""
        T = matrix44.create_from_translation(self.position)
        R = matrix44.create_from_eulers(self.rotation_euler)  # euler: [rx, ry, rz] radians
        S = matrix44.create_from_scale(self.scale)
        # combine: T * R * S
        TR = matrix44.multiply(T, R)
        TRS = matrix44.multiply(TR, S)
        return TRS

    def get_world_matrix(self):
        """Return world matrix by walking up the parents (4x4 numpy array)"""
        local = self.get_local_matrix()
        if self.parent is None:
            return local
        # parent's world * local
        parent_world = self.parent.get_world_matrix()
        return matrix44.multiply(parent_world, local)

    def traverse(self):
        """Yield self and all descendants (pre-order)"""
        yield self
        for c in self.children:
            yield from c.traverse()


# ---------------------------
# Mesh node (holds vertices in local space)
# ---------------------------
class Mesh(Object3D):
    def __init__(self, vertices: np.ndarray, name="Mesh"):
        """
        vertices: (N,3) numpy array in local space
        """
        super().__init__(name=name)
        assert isinstance(vertices, np.ndarray) and vertices.ndim == 2 and vertices.shape[1] == 3
        self.vertices = vertices.copy()


# ---------------------------
# Camera (perspective)
# ---------------------------
class PerspectiveCamera(Object3D):
    def __init__(self, fov_y_deg=60.0, aspect=1.0, near=0.1, far=100.0, name="Camera"):
        super().__init__(name=name)
        self.fov_y_deg = float(fov_y_deg)
        self.aspect = float(aspect)
        self.near = float(near)
        self.far = float(far)

    def get_view_matrix(self):
        """View = inverse(world)"""
        world = self.get_world_matrix()
        inv = matrix44.inverse(world)
        return inv

    def get_projection_matrix(self):
        """Create projection matrix using pyrr.matrix44 (fov in degrees)."""
        # matrix44.create_perspective_projection(fovy, aspect, near, far)
        return matrix44.create_perspective_projection(self.fov_y_deg, self.aspect, self.near, self.far)


# ---------------------------
# Scene + renderer
# ---------------------------
class Scene:
    def __init__(self):
        self.root = Object3D(name="Root")

    def add(self, obj: Object3D):
        self.root.add(obj)

    def render(self, camera: PerspectiveCamera, figsize=(7, 7), show_axes=True, annotate=False):
        """
        Projects all Mesh vertices with the given camera and plots them in NDC (-1..1).
        - camera: PerspectiveCamera instance (position/orientation define view)
        - This renderer does no hidden-surface removal; it simply projects points.
        """
        # collect all meshes
        meshes = [node for node in self.root.traverse() if isinstance(node, Mesh)]
        if len(meshes) == 0:
            print("Scene.render: no Mesh nodes to render.")
            return

        view = camera.get_view_matrix()  # 4x4
        proj = camera.get_projection_matrix()  # 4x4

        plt.figure(figsize=figsize)
        ax = plt.gca()

        for m_idx, mesh in enumerate(meshes):
            # world matrix for the mesh
            world = mesh.get_world_matrix()
            # transform vertices to world -> view -> clip
            pts_world_h = to_homogeneous(mesh.vertices)  # (N,4)
            # apply view: row-vector convention -> points @ view.T
            pts_view_h = pts_world_h @ view.T  # (N,4)
            # apply projection
            pts_clip_h = pts_view_h @ proj.T  # (N,4)

            # perspective divide (handle w=0 safely)
            w = pts_clip_h[:, 3:4]
            # avoid division by zero: mark invalid as large values
            eps = 1e-8
            w_safe = np.where(np.abs(w) < eps, np.sign(w) * eps + eps, w)
            pts_ndc = pts_clip_h[:, :3] / w_safe

            x = pts_ndc[:, 0]
            y = pts_ndc[:, 1]
            z = pts_ndc[:, 2]  # not used for drawing, but could be used for depth sorting

            # Only draw points inside clip space (w>0 and |ndc|<=1 in x/y/z)
            inside = (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0) & (np.abs(z) <= 1.0) & (w[:, 0] > 0)

            # plot visible points and optionally annotate
            ax.scatter(x[inside], y[inside], s=200, label=f"{mesh.name} (mesh {m_idx})")
            if annotate:
                for i_pt, (xx, yy) in enumerate(zip(x[inside], y[inside])):
                    ax.text(xx, yy, f"{mesh.name}:{i_pt}", fontsize=6)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("NDC X")
        ax.set_ylabel("NDC Y")
        ax.set_title("Scene projection (NDC)")
        if not show_axes:
            ax.set_axis_off()
        ax.invert_yaxis()  # match screen coordinates (optional)
        ax.grid(True)
        ax.legend()
        plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Create a scene
    scene = Scene()

    # Create a simple square mesh in the XY plane centered at origin (local space)
    square = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=float,
    )

    mesh_root = Mesh(square, name="SquareRoot")
    mesh_root.position = vec3(0.0, 0.0, 5.0)  # move it 5 units forward in z
    mesh_root.rotation_euler = vec3(0.0, 0.0, 0.2)  # small rotation around z
    mesh_root.scale = vec3(2.0, 1.0, 1.0)  # non-uniform scale

    # child mesh (offset from parent)
    mesh_child = Mesh(square * 0.3, name="SquareChild")
    mesh_child.position = vec3(1.0, 0.8, 0.0)  # local offset relative to parent
    mesh_child.rotation_euler = vec3(0.0, 0.6, 0.0)
    mesh_root.add(mesh_child)

    # add to scene root
    scene.add(mesh_root)

    # camera: positioned at origin, looking down -Z by default (since no lookAt is explicitly set)
    cam = PerspectiveCamera(fov_y_deg=60.0, aspect=1.0, near=0.1, far=100.0, name="MainCamera")
    cam.position = vec3(0.0, 0.0, 0)  # put camera at origin
    cam.rotation_euler = vec3(0.0, 0.0, 0.0)  # no rotation (camera looks down -Z)

    # Optionally, tilt/translate camera:
    # cam.position = vec3(0.0, -0.5, -0.5)
    # cam.rotation_euler = vec3(0.12, 0.0, 0.0)

    # Render
    scene.render(cam, figsize=(6, 6), show_axes=True, annotate=True)
