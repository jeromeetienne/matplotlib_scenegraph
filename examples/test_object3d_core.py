import unittest
import numpy as np
from pyrr import matrix44, quaternion, vector3

from mpl_graph.core.object_3d import Object3D


class TestObject3DCore(unittest.TestCase):
    def test_defaults(self):
        o = Object3D()
        # position, scale, rotation quaternion
        np.testing.assert_allclose(o.position, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(o.scale, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        # identity quaternion [x, y, z, w] = [0,0,0,1]
        self.assertEqual(o.rotation.shape, (4,))
        np.testing.assert_allclose(o.rotation, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), rtol=1e-6, atol=1e-6)

        # matrices default to identity
        np.testing.assert_allclose(o.get_local_matrix(), matrix44.create_identity(dtype=np.float32))
        np.testing.assert_allclose(o.get_world_matrix(), matrix44.create_identity(dtype=np.float32))

    def test_local_matrix_srt_order(self):
        o = Object3D()
        o.position = vector3.create(1.0, 2.0, 3.0)
        o.scale = vector3.create(2.0, 3.0, 4.0)
        o.rotation = quaternion.create(dtype=np.float32)  # identity quat

        o.update_local_matrix()

        # Build expected matrix using the same order: scale -> rotate -> translate
        s = matrix44.create_from_scale(o.scale, dtype=np.float32)
        r = matrix44.create_from_quaternion(o.rotation, dtype=np.float32)
        t = matrix44.create_from_translation(o.position, dtype=np.float32)

        expected = matrix44.create_identity(dtype=np.float32)
        expected = matrix44.multiply(expected, s)
        expected = matrix44.multiply(expected, r)
        expected = matrix44.multiply(expected, t)

        np.testing.assert_allclose(o.get_local_matrix(), expected, rtol=1e-6, atol=1e-6)

    def test_parent_child_world_matrix(self):
        parent = Object3D()
        parent.position = vector3.create(10.0, 0.0, 0.0)
        parent.rotation = quaternion.create_from_z_rotation(np.pi / 2.0, dtype=np.float32)  # 90 degrees around Z
        parent.scale = vector3.create(1.0, 1.0, 1.0)

        child = Object3D()
        child.position = vector3.create(1.0, 0.0, 0.0)
        child.rotation = quaternion.create(dtype=np.float32)
        child.scale = vector3.create(1.0, 1.0, 1.0)

        parent.add(child)
        parent.update_world_matrix()

        # Build expected matrices with class' multiplication order (local first, then parent)
        s_p = matrix44.create_from_scale(parent.scale, dtype=np.float32)
        r_p = matrix44.create_from_quaternion(parent.rotation, dtype=np.float32)
        t_p = matrix44.create_from_translation(parent.position, dtype=np.float32)
        local_p = matrix44.create_identity(dtype=np.float32)
        local_p = matrix44.multiply(local_p, s_p)
        local_p = matrix44.multiply(local_p, r_p)
        local_p = matrix44.multiply(local_p, t_p)

        s_c = matrix44.create_from_scale(child.scale, dtype=np.float32)
        r_c = matrix44.create_from_quaternion(child.rotation, dtype=np.float32)
        t_c = matrix44.create_from_translation(child.position, dtype=np.float32)
        local_c = matrix44.create_identity(dtype=np.float32)
        local_c = matrix44.multiply(local_c, s_c)
        local_c = matrix44.multiply(local_c, r_c)
        local_c = matrix44.multiply(local_c, t_c)

        expected_parent_world = local_p.copy()
        expected_child_world = matrix44.create_identity(dtype=np.float32)
        expected_child_world = matrix44.multiply(expected_child_world, local_c)
        expected_child_world = matrix44.multiply(expected_child_world, expected_parent_world)

        np.testing.assert_allclose(parent.get_world_matrix(), expected_parent_world, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(child.get_world_matrix(), expected_child_world, rtol=1e-6, atol=1e-6)

        # World position is last row, first 3 components in this convention
        np.testing.assert_allclose(parent.get_world_position(), expected_parent_world[3, :3], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(child.get_world_position(), expected_child_world[3, :3], rtol=1e-6, atol=1e-6)

    def test_world_quaternion_extraction(self):
        o = Object3D()
        o.rotation = quaternion.create_from_y_rotation(np.pi / 3.0, dtype=np.float32)  # 60 deg around Y
        o.update_world_matrix()

        q_world = o.get_world_rotation_quaternion()

        # Expected quaternion equals local rotation since no parent and uniform scale
        # Comparison normalized to avoid small numeric differences
        def _norm(q):
            q = np.asarray(q, dtype=np.float32)
            return q / np.linalg.norm(q)

        np.testing.assert_allclose(_norm(q_world), _norm(o.rotation), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
