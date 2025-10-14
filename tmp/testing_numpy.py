import numpy as np


def test_numpy():
    arr = np.array([1, 2, 3])
    return arr


a = test_numpy()
b = test_numpy()

print("a:", a)
print("b:", b)
a[0] = 10
print("a after modification:", a)
print("b after a modification (should be unchanged):", b)
