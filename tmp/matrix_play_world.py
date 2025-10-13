import numpy as np
from pyrr import matrix44

scale = matrix44.create_from_scale([2.0, 2.0, 2.0])
# scale = matrix44.create_from_scale([1.0, 1.0, 1.0])
rot = matrix44.create_from_y_rotation(np.pi / 2)
# rot = matrix44.create_from_x_rotation(np.pi / 2)
# trans = matrix44.create_from_translation([0.0, 1.0, 0.0])
trans = matrix44.create_from_translation([0.0, 1.0, 0.0])

local = matrix44.create_identity()
local = matrix44.multiply(local, scale)
local = matrix44.multiply(local, rot)
local = matrix44.multiply(local, trans)
# local = matrix44.multiply(scale, local)
# local = matrix44.multiply(rot, local)
# local = matrix44.multiply(trans, local)
# local = trans @ rot @ scale
local = scale @ rot @ trans

print("done")
print(local)
# expected answer is [0, 2, 1]
print(np.array([1.0, 0.0, 0.0, 1.0]) @ local)
