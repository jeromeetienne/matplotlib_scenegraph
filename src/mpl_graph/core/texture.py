import numpy as np


class Texture:
    __slots__ = (
        "data",
        "extent",
    )

    def __init__(self, data: np.ndarray) -> None:
        assert data.ndim == 3 and data.shape[2] in [3, 4], "image should be of shape [H, W, 3] or [H, W, 4]"

        self.data: np.ndarray = data

    def width(self) -> int:
        """Return the width of the texture in pixels."""
        return self.data.shape[1]

    def height(self) -> int:
        """Return the height of the texture in pixels."""
        return self.data.shape[0]

    def aspect_ratio(self) -> float:
        """Return the aspect ratio (width / height) of the texture."""
        return self.width() / self.height()
