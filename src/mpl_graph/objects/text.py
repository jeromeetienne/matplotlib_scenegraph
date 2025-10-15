# pip imports
from pyrr import vector3
import numpy as np

# Local imports
from ..core.constants import Constants
from ..core.object_3d import Object3D
from ..core.texture import Texture
from ..materials import TextMaterial


class Text(Object3D):
    __slots__ = ("content", "material")

    def __init__(self, content: str | None = None, material: TextMaterial | None = None) -> None:
        super().__init__()

        self.name = f"a {Text.__name__}"

        self.content: str = content if content is not None else "Text"
        """Content of the text."""
        self.material: TextMaterial = material if material is not None else TextMaterial()
        """Material of the text."""
