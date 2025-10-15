import unittest
import numpy as np
from pyrr import quaternion

from mpl_graph.core.object_3d import Object3D


def norm(q):
    q = np.asarray(q, dtype=np.float32)
    return q / np.linalg.norm(q)


class TestObject3DRotations(unittest.TestCase):
    def test_rotate_axes(self):
        o = Object3D()
        # Start with identity
        np.testing.assert_allclose(o.rotation, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # Rotate 90 deg around X
        o.rotate_x(np.pi / 2)
        qx = quaternion.create_from_x_rotation(np.pi / 2, dtype=np.float32)
        if np.dot(norm(o.rotation), norm(qx)) < 0:
            qx = -qx
        np.testing.assert_allclose(norm(o.rotation), norm(qx), rtol=1e-5, atol=1e-5)

        # Then 45 deg around Y (composition: qy * qx)
        o.rotate_y(np.pi / 4)
        qy = quaternion.create_from_y_rotation(np.pi / 4, dtype=np.float32)
        q_expected = quaternion.normalize(quaternion.cross(qy, qx))
        if np.dot(norm(o.rotation), norm(q_expected)) < 0:
            q_expected = -q_expected
        np.testing.assert_allclose(norm(o.rotation), norm(q_expected), rtol=1e-5, atol=1e-5)

        # Finally 30 deg around Z
        o.rotate_z(np.deg2rad(30.0))
        qz = quaternion.create_from_z_rotation(np.deg2rad(30.0), dtype=np.float32)
        q_expected2 = quaternion.normalize(quaternion.cross(qz, q_expected))
        if np.dot(norm(o.rotation), norm(q_expected2)) < 0:
            q_expected2 = -q_expected2
        np.testing.assert_allclose(norm(o.rotation), norm(q_expected2), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
