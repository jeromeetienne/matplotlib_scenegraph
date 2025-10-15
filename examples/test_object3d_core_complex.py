import unittest
import numpy as np
from pyrr import matrix44, quaternion, vector3

from mpl_graph.core.object_3d import Object3D


def build_local(position: np.ndarray, scale: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    s = matrix44.create_from_scale(scale, dtype=np.float32)
    r = matrix44.create_from_quaternion(rotation, dtype=np.float32)
    t = matrix44.create_from_translation(position, dtype=np.float32)
    m = matrix44.create_identity(dtype=np.float32)
    m = matrix44.multiply(m, s)
    m = matrix44.multiply(m, r)
    m = matrix44.multiply(m, t)
    return m


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / np.linalg.norm(q)


def extract_quat_from_world(world: np.ndarray) -> np.ndarray:
    # Remove scale from the 3x3 part by normalizing row vectors
    m = world.astype(np.float64)
    x = m[0, :3]
    y = m[1, :3]
    z = m[2, :3]
    if np.linalg.norm(x) > 0:
        x = x / np.linalg.norm(x)
    if np.linalg.norm(y) > 0:
        y = y / np.linalg.norm(y)
    if np.linalg.norm(z) > 0:
        z = z / np.linalg.norm(z)
    R = np.stack([x, y, z], axis=0)

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
    return normalize_quat(q)


class TestObject3DCoreComplex(unittest.TestCase):
    def test_deep_hierarchy_mixed_transforms(self):
        # Build a 4-level chain: A -> B -> C -> D
        A = Object3D()
        A.name = "A"
        B = Object3D()
        B.name = "B"
        C = Object3D()
        C.name = "C"
        D = Object3D()
        D.name = "D"

        # Assign transforms (mix rotations/scales/translations)
        A.position = vector3.create(3.0, -2.0, 5.0)
        A.scale = vector3.create(1.5, 1.0, 0.5)
        A.rotation = quaternion.create_from_x_rotation(np.deg2rad(25), dtype=np.float32)

        B.position = vector3.create(-4.0, 1.0, 0.0)
        B.scale = vector3.create(0.75, 2.0, 1.25)
        B.rotation = quaternion.create_from_y_rotation(np.deg2rad(-40), dtype=np.float32)

        C.position = vector3.create(0.0, 3.0, 2.0)
        C.scale = vector3.create(1.0, 1.0, 1.0)
        C.rotation = quaternion.create_from_z_rotation(np.deg2rad(90), dtype=np.float32)

        D.position = vector3.create(1.0, 0.0, -1.0)
        D.scale = vector3.create(2.0, 0.5, 3.0)
        D.rotation = quaternion.create_from_y_rotation(np.deg2rad(15), dtype=np.float32)

        # Link
        A.add_child(B)
        B.add_child(C)
        C.add_child(D)

        # Update
        A.update_world_matrix()

        # Build expected using the same multiplication order as implementation
        L_A = build_local(A.position, A.scale, A.rotation)
        W_A = L_A.copy()

        L_B = build_local(B.position, B.scale, B.rotation)
        W_B = matrix44.multiply(matrix44.create_identity(dtype=np.float32), L_B)
        W_B = matrix44.multiply(W_B, W_A)

        L_C = build_local(C.position, C.scale, C.rotation)
        W_C = matrix44.multiply(matrix44.create_identity(dtype=np.float32), L_C)
        W_C = matrix44.multiply(W_C, W_B)

        L_D = build_local(D.position, D.scale, D.rotation)
        W_D = matrix44.multiply(matrix44.create_identity(dtype=np.float32), L_D)
        W_D = matrix44.multiply(W_D, W_C)

        # Assertions
        np.testing.assert_allclose(A.get_world_matrix(), W_A, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(B.get_world_matrix(), W_B, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(C.get_world_matrix(), W_C, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(D.get_world_matrix(), W_D, rtol=1e-5, atol=1e-5)

        # World position should match last row
        np.testing.assert_allclose(D.get_world_position(), W_D[3, :3], rtol=1e-6, atol=1e-6)

        # Quaternion extraction should match world rotation regardless of scales
        q_world_D = normalize_quat(D.get_world_rotation_quaternion())
        q_expected_D = extract_quat_from_world(W_D)
        # Compare quaternion equivalence accounting for double-cover (q ~ -q)
        if np.dot(q_world_D, q_expected_D) < 0:
            q_world_D = -q_world_D
        np.testing.assert_allclose(q_world_D, q_expected_D, rtol=1e-4, atol=1e-4)

    def test_non_uniform_scale_and_rotation_quaternion_integrity(self):
        o = Object3D()
        o.scale = vector3.create(5.0, 0.25, 2.0)
        o.rotation = quaternion.create_from_eulers([0.2, -0.7, 1.1], dtype=np.float32)
        o.position = vector3.create(-2.0, 4.0, 3.0)
        o.update_world_matrix()

        q_world = normalize_quat(o.get_world_rotation_quaternion())
        q_local = normalize_quat(o.rotation)
        # world = local when no parent (scale shouldn't affect extracted rotation)
        if np.dot(q_world, q_local) < 0:
            q_world = -q_world
        np.testing.assert_allclose(q_world, q_local, rtol=1e-5, atol=1e-5)

    def test_traverse_root_and_remove(self):
        root = Object3D()
        root.name = "root"
        a = Object3D()
        a.name = "a"
        b = Object3D()
        b.name = "b"
        c = Object3D()
        c.name = "c"
        root.add_child(a)
        root.add_child(b)
        a.add_child(c)

        names = [n.name for n in root.traverse()]
        # Depth-first order with parent first as implemented
        self.assertEqual(names, ["root", "a", "c", "b"])
        self.assertIs(root.root(), root)
        self.assertIs(c.root(), root)

        # Remove node and ensure traverse updates
        a.remove_child(c)
        names2 = [n.name for n in root.traverse()]
        self.assertEqual(names2, ["root", "a", "b"])

        # After detach, child's world should equal local once updated on its own
        c.update_world_matrix()  # no parent
        np.testing.assert_allclose(c.get_world_matrix(), c.get_local_matrix(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
