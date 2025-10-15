# pip imports
from pyrr import vector3, matrix44, quaternion
import numpy as np
from typing import Callable
from typing import Protocol

# local imports
from .random import Random
from .event import Event


# We can define the expected function signature using a Protocol for clarity.
class PreRenderingCallback(Protocol):
    def __call__(self, object_3d: "Object3D", camera: "CameraBase") -> None: ...  # type: ignore


class PostTransformCallback(Protocol):
    def __call__(self, vertices_transformed: np.ndarray) -> None: ...  # type: ignore


class PostRenderingCallback(Protocol):
    def __call__(self) -> None: ...  # type: ignore


class Object3D:
    __slots__ = (
        "uuid",
        "name",
        "position",
        "rotation",
        "scale",
        "parent",
        "_children",
        "_local_matrix",
        "_world_matrix",
        "pre_rendering",
        "post_transform",
        "post_rendering",
    )

    def __init__(self) -> None:
        self.uuid = Random.random_uuid()
        self.name = f"a {Object3D.__name__}"

        self.position = vector3.create(0.0, 0.0, 0.0)
        """Position vector (x, y, z) in local space."""
        self.rotation = quaternion.create(dtype=np.float32)  # Identity quaternion (x, y, z, w)
        """Rotation as a quaternion (x, y, z, w) in local space."""
        self.scale = vector3.create(1.0, 1.0, 1.0)
        """Scale vector (sx, sy, sz) in local space."""

        self.parent: Object3D | None = None
        """Parent Object3D instance or None."""
        self._children: list[Object3D] = []
        """List of child Object3D instances."""

        self._local_matrix = matrix44.create_identity(dtype=np.float32)
        self._world_matrix = matrix44.create_identity(dtype=np.float32)

        self.pre_rendering = Event[PreRenderingCallback]()
        """Event triggered before rendering the visual."""

        self.post_transform = Event[PostTransformCallback]()
        """
        Event triggered after applying 3d transformations to the visual.

        Arguments sent to subscribers:
        - renderer: The renderer instance performing the rendering.
        - camera: The camera used for rendering.
        - transformed_positions: The numpy array of transformed positions (shape: n x 3).
        """

        self.post_rendering = Event[PostRenderingCallback]()
        """Event triggered after rendering the visual."""

    # =============================================================================
    # add/remove child
    # =============================================================================
    def add(self, child: "Object3D") -> None:
        """Add a child Object3D to this object."""
        child.parent = self
        self._children.append(child)

    def remove(self, child: "Object3D") -> None:
        """Remove a child Object3D from this object."""
        assert child in self._children, "Child not found"
        self._children.remove(child)
        child.parent = None

    def traverse(self) -> list["Object3D"]:
        """Return a list of this object and all its descendants (children, grandchildren, etc.)."""
        objects: list[Object3D] = [self]
        for child in self._children:
            child_objects = child.traverse()
            objects.extend(child_objects)
        return objects

    def root(self) -> "Object3D":
        """Return the root Object3D of the scene graph this object belongs to."""
        object: Object3D = self
        while object.parent is not None:
            object = object.parent
        return object

    # =============================================================================
    # Update matrix
    # =============================================================================
    def update_local_matrix(self) -> None:
        scale_matrix = matrix44.create_from_scale(self.scale, dtype=np.float32)
        rotation_matrix = matrix44.create_from_quaternion(self.rotation, dtype=np.float32)
        translation_matrix = matrix44.create_from_translation(self.position, dtype=np.float32)

        # self._local_matrix = trans_m @ rot_m @ scale_m
        # self._local_matrix = scale_matrix @ rotation_matrix @ translation_matrix

        # compute the local matrix: first `scale`, then `rotate`, then `translate`
        self._local_matrix = matrix44.create_identity(dtype=np.float32)
        self._local_matrix = matrix44.multiply(self._local_matrix, scale_matrix)
        self._local_matrix = matrix44.multiply(self._local_matrix, rotation_matrix)
        self._local_matrix = matrix44.multiply(self._local_matrix, translation_matrix)

    def update_world_matrix(self, parent_world_matrix: np.ndarray | None = None) -> None:
        self.update_local_matrix()

        if parent_world_matrix is not None:
            # Compute world matrix by combining local and parent world matrices: first `local` then `parent's world`
            self._world_matrix = matrix44.create_identity(dtype=np.float32)
            self._world_matrix = matrix44.multiply(self._world_matrix, self._local_matrix)
            self._world_matrix = matrix44.multiply(self._world_matrix, parent_world_matrix)
        else:
            self._world_matrix = self._local_matrix.copy()

        for child in self._children:
            child.update_world_matrix(self._world_matrix)

    def get_local_matrix(self) -> np.ndarray:
        # self.update_local_matrix()
        return self._local_matrix

    def get_world_matrix(self) -> np.ndarray:
        # self.update_world_matrix()
        return self._world_matrix

    # =============================================================================
    # get_world_position/scale/rotation (quaternion)
    # =============================================================================
    def get_world_position(self) -> np.ndarray:
        return self._world_matrix[3, :3]

    def get_world_scale(self) -> np.ndarray:
        sx = np.linalg.norm(self._world_matrix[0, :3])
        sy = np.linalg.norm(self._world_matrix[1, :3])
        sz = np.linalg.norm(self._world_matrix[2, :3])
        return vector3.create(sx, sy, sz)  # type: ignore

    def get_world_rotation_quaternion(self) -> np.ndarray:
        """Return world rotation as a normalized quaternion.

        Note: extracts rotation from the world matrix by removing the scale.
        """
        # Extract and normalize the rotation axes to remove scale
        m = self._world_matrix.astype(np.float64)
        x_axis = m[0, :3]
        y_axis = m[1, :3]
        z_axis = m[2, :3]

        sx = np.linalg.norm(x_axis)
        sy = np.linalg.norm(y_axis)
        sz = np.linalg.norm(z_axis)

        if sx > 0:
            x_axis = x_axis / sx
        if sy > 0:
            y_axis = y_axis / sy
        if sz > 0:
            z_axis = z_axis / sz

        R = np.stack([x_axis, y_axis, z_axis], axis=0)

        # Convert 3x3 rotation matrix to quaternion (x, y, z, w)
        trace = np.trace(R)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        q = np.array([qx, qy, qz, qw], dtype=np.float32)
        # Normalize to ensure a valid quaternion
        q = q / np.linalg.norm(q)
        return q

    # =============================================================================
    # rotate_x / rotate_y / rotate_z (local space)
    # =============================================================================
    def rotate_x(self, angle_rad: float) -> "Object3D":
        """Rotate locally around +X by angle_rad (radians)."""
        rot = quaternion.create_from_x_rotation(float(angle_rad), dtype=np.float32)
        self.rotation = quaternion.normalize(quaternion.cross(rot, self.rotation))
        return self

    def rotate_y(self, angle_rad: float) -> "Object3D":
        """Rotate locally around +Y by angle_rad (radians)."""
        rot = quaternion.create_from_y_rotation(float(angle_rad), dtype=np.float32)
        self.rotation = quaternion.normalize(quaternion.cross(rot, self.rotation))
        return self

    def rotate_z(self, angle_rad: float) -> "Object3D":
        """Rotate locally around +Z by angle_rad (radians)."""
        rot = quaternion.create_from_z_rotation(float(angle_rad), dtype=np.float32)
        self.rotation = quaternion.normalize(quaternion.cross(rot, self.rotation))
        return self

    def reset_rotation(self) -> "Object3D":
        """Reset local rotation to identity (no rotation)."""
        self.rotation = quaternion.create(dtype=np.float32)
        return self

    def look_at(self, target: np.ndarray, up: np.ndarray | None = None) -> "Object3D":
        """Orient this object so its forward (-Z) points toward `target`.

        Uses pyrr's look-at to derive orientation and minimizes custom math. Right-handed, forward is -Z.
        """
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        eye = np.asarray(self.position, dtype=np.float32)
        tgt = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        # If target equals eye, do nothing
        if np.allclose(eye, tgt):
            return self

        # View matrix (world -> view)
        view = matrix44.create_look_at(eye=eye, target=tgt, up=up, dtype=np.float32)
        # Extract view rotation and invert it to get object rotation (view is R^T for orthonormal)
        R_view = view[:3, :3]
        R_obj = R_view.T

        # Prefer pyrr for matrix->quaternion if available; otherwise fallback
        q = quaternion.create_from_matrix(R_obj.astype(np.float32), dtype=np.float32)

        self.rotation = q / np.linalg.norm(q)
        return self
