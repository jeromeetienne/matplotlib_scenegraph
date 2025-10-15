# quaternion-based Object3D (no eulers)
from pyrr import vector3, matrix44
from pyrr import quaternion as quat
import numpy as np
from typing import Callable
import math
from typing import Protocol

# local imports (kept from your original file)
from .random import Random
from .event import Event


class PreRenderingCallback(Protocol):
    def __call__(self, object_3d: "Object3D", camera: "CameraBase") -> None: ...  # type: ignore


class PostTransformCallback(Protocol):
    def __call__(self, vertices_transformed: np.ndarray) -> None: ...  # type: ignore


class PostRenderingCallback(Protocol):
    def __call__(self) -> None: ...  # type: ignore


# Helper: robust quaternion multiply (try to use quat.cross, fallback to Hamilton product)
_try_use_quat_cross = hasattr(quat, "cross")


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Quaternion multiply: returns a * b (apply b, then a) using either quat.cross (if available)
    or an internal Hamilton product implementation.
    Quaternions are expected as [x, y, z, w].
    """
    if _try_use_quat_cross:
        # user requested using quat.cross()
        return quat.cross(a, b)
    # fallback: Hamilton product (x,y,z,w)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.array([x, y, z, w], dtype=np.float32)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _quaternion_from_rotation_matrix(m3: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix (numpy array) into quaternion [x,y,z,w].
    Implementation is numerically stable standard algorithm.
    """
    m = np.asarray(m3, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    return _normalize_quat(q)


class Object3D:
    __slots__ = (
        "uuid",
        "name",
        "position",
        "rotation_quaternion",
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

        # quaternion as [x, y, z, w]
        self.rotation_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        """Rotation quaternion [x, y, z, w] in local space."""

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
    def add_child(self, child: "Object3D") -> None:
        child.parent = self
        self._children.append(child)

    def remove_child(self, child: "Object3D") -> None:
        assert child in self._children, "Child not found"
        self._children.remove(child)
        child.parent = None

    def traverse(self) -> list["Object3D"]:
        objects: list[Object3D] = [self]
        for child in self._children:
            child_objects = child.traverse()
            objects.extend(child_objects)
        return objects

    def root(self) -> "Object3D":
        object: Object3D = self
        while object.parent is not None:
            object = object.parent
        return object

    # =============================================================================
    # Update matrix
    # =============================================================================
    def update_local_matrix(self) -> None:
        """
        Build local matrix from scale, quaternion rotation, translation.
        Order: scale -> rotate -> translate (as in your original implementation).
        """
        scale_matrix = matrix44.create_from_scale(self.scale, dtype=np.float32)

        q = _normalize_quat(self.rotation_quaternion)
        # pyrr.matrix44.create_from_quaternion expects quaternion as [x,y,z,w]
        rotation_matrix = matrix44.create_from_quaternion(q, dtype=np.float32)
        translation_matrix = matrix44.create_from_translation(self.position, dtype=np.float32)

        # local = I * scale * rotation * translation
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
        return self._local_matrix

    def get_world_matrix(self) -> np.ndarray:
        return self._world_matrix

    # =============================================================================
    # get_world_position/scale/quaternion
    # =============================================================================
    def get_world_position(self) -> np.ndarray:
        # matches your convention: translation in row 3
        return self._world_matrix[3, :3]

    def get_world_scale(self) -> np.ndarray:
        sx = np.linalg.norm(self._world_matrix[0, :3])
        sy = np.linalg.norm(self._world_matrix[1, :3])
        sz = np.linalg.norm(self._world_matrix[2, :3])
        return vector3.create(sx, sy, sz)  # type: ignore

    def get_world_quaternion(self) -> np.ndarray:
        """
        Compute world quaternion from world 3x3 rotation part.
        Returns quaternion as [x, y, z, w].
        """
        rot3 = self._world_matrix[:3, :3]
        return _quaternion_from_rotation_matrix(rot3)

    # =============================================================================
    # Quaternion helpers
    # =============================================================================
    def set_quaternion(self, q: np.ndarray) -> None:
        self.rotation_quaternion = _normalize_quat(q)

    def rotate_axis(self, axis: np.ndarray, angle_rad: float) -> None:
        """
        Rotate the object *in local space* around axis (3,) by angle_rad.
        Creates an axis-angle quaternion then multiplies it with current quaternion.
        Multiplication order: new_q * current_q (apply current, then new)
        """
        axis = np.asarray(axis, dtype=np.float32)
        if np.linalg.norm(axis) == 0:
            return
        axis = axis / np.linalg.norm(axis)
        # pyrr.quaternion.create_from_axis_rotation returns [x,y,z,w]
        q_new = quat.create_from_axis_rotation(axis, float(angle_rad))
        q_new = _normalize_quat(q_new)
        # combine: q = q_new * q_current (apply q_current then q_new)
        q_combined = _quat_multiply(q_new, self.rotation_quaternion)
        self.rotation_quaternion = _normalize_quat(q_combined)

    def look_at(self, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)) -> None:
        """
        Orient the object so that its -Z axis faces `target`, following three.js lookAt semantics.
        `up` defaults to world +Y. Both `target` and `up` are 3-element vectors.
        """
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        pos = np.asarray(self.position, dtype=np.float32)

        # forward (target - position)
        forward = target - pos
        if np.linalg.norm(forward) == 0:
            # no-op if target == position
            return
        forward = forward / np.linalg.norm(forward)

        # three.js defines the object's "forward" as -Z, so zaxis = -forward
        z = -forward
        # x = up × z
        x = np.cross(up, z)
        if np.linalg.norm(x) == 0:
            # up and z are parallel — choose an arbitrary orthogonal up slightly
            if abs(up[2]) < 0.9:
                up_candidate = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                up_candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            x = np.cross(up_candidate, z)
        x = x / np.linalg.norm(x)

        # y = z × x
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)

        # build 3x3 rotation matrix where rows correspond to basis vectors (x, y, z),
        # matching the convention used elsewhere in this class.
        rot3 = np.zeros((3, 3), dtype=np.float32)
        rot3[0, :3] = x
        rot3[1, :3] = y
        rot3[2, :3] = z

        q = _quaternion_from_rotation_matrix(rot3)
        self.rotation_quaternion = _normalize_quat(q)

    # =============================================================================
    # string repr (optional)
    # =============================================================================
    def __repr__(self) -> str:
        return f"<Object3D name={self.name} uuid={self.uuid}>"
