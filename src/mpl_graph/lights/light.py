# local imports
from ..core.object_3d import Object3D


class Light(Object3D):
    def __init__(self):
        super().__init__()

        self.name = f"a {Light.__name__}"
