import unittest
import numpy as np
from pyrr import matrix44

from mpl_graph.core.object_3d import Object3D


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


class TestObject3DLookAt(unittest.TestCase):
    def test_look_at_basic_forward_alignment(self):
        o = Object3D()
        o.position[:] = [0.0, 0.0, 0.0]
        target = np.array([0.0, 0.0, -5.0], dtype=np.float32)

        o.look_at(target)

        R = matrix44.create_from_quaternion(o.rotation, dtype=np.float32)
        forward_from_rot = -R[2, :3]  # -Z axis should face target
        expected_forward = normalize(target - o.position)

        np.testing.assert_allclose(forward_from_rot, expected_forward, rtol=1e-5, atol=1e-5)

    def test_look_at_arbitrary_point(self):
        o = Object3D()
        o.position[:] = [1.0, 2.0, 3.0]
        target = np.array([4.0, -2.0, 5.0], dtype=np.float32)

        o.look_at(target)

        R = matrix44.create_from_quaternion(o.rotation, dtype=np.float32)
        forward_from_rot = -R[2, :3]
        expected_forward = normalize(target - o.position)
        np.testing.assert_allclose(forward_from_rot, expected_forward, rtol=1e-5, atol=1e-5)

    def test_look_at_with_custom_up(self):
        o = Object3D()
        o.position[:] = [0.0, 0.0, 0.0]
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Z-up

        o.look_at(target, up)

        R = matrix44.create_from_quaternion(o.rotation, dtype=np.float32)
        forward_from_rot = -R[2, :3]
        expected_forward = normalize(target - o.position)
        np.testing.assert_allclose(forward_from_rot, expected_forward, rtol=1e-5, atol=1e-5)

        # Y axis from rotation should be reasonably aligned with provided up (not flipped)
        y_axis = R[1, :3]
        dot_up = float(np.dot(normalize(y_axis), normalize(up)))
        self.assertGreater(dot_up, 0.7)

    def test_look_at_degenerate_same_position(self):
        o = Object3D()
        o.position[:] = [2.0, 3.0, 4.0]
        before = o.rotation.copy()
        o.look_at(o.position.copy())  # target == position
        np.testing.assert_allclose(o.rotation, before, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
