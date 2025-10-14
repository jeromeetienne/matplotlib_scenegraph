from ..core.object_3d import Object3D


class Scene(Object3D):
    def __init__(self) -> None:
        super().__init__()
        self.name = "the scene"
