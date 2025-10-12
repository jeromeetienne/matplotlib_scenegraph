import numpy as np

class Texture:
    __slots__ = ("image", "extent",)

    def __init__(self, image: np.ndarray) -> None:
        assert image.ndim == 3 and image.shape[2] in [3, 4], "image should be of shape [H, W, 3] or [H, W, 4]"

        self.image: np.ndarray = image

