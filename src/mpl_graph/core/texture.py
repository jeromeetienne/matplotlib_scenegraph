# pip imports
import numpy as np
import matplotlib.image


class Texture:
    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        """
        float texture image data of shape [H, W, 3] or [H, W, 4] in range [0, 1]
        """

        assert data.ndim == 3 and data.shape[2] in [3, 4], f"image should be of shape [H, W, 3] or [H, W, 4], got {data.shape}"
        assert data.dtype in [np.float32, np.float64], f"image should be of type float32 or float64, got {data.dtype}"

        self.data: np.ndarray = data

    def copy(self) -> "Texture":
        """Return a copy of the texture."""
        return Texture(self.data.copy())

    def width(self) -> int:
        """Return the width of the texture in pixels."""
        return self.data.shape[1]

    def height(self) -> int:
        """Return the height of the texture in pixels."""
        return self.data.shape[0]

    def aspect_ratio(self) -> float:
        """Return the aspect ratio of the texture (width / height)."""
        return self.width() / self.height()

    def has_alpha(self) -> bool:
        """Return True if the texture has an alpha channel."""
        return self.data.shape[2] == 4

    def strip_alpha(self) -> "Texture":
        """Strip the texture of the alpha channel in place."""

        if not self.has_alpha():
            return self

        self.data = self.data[:, :, :3]

        return self

    @staticmethod
    def from_file(file_path: str) -> "Texture":
        """
        Load a texture image from file.
        """

        # read image using matplotlib
        texture_data = matplotlib.image.imread(file_path)

        if texture_data.dtype == np.uint8:
            # convert to float32 in range [0, 1]
            texture_data = texture_data.astype(np.float32) / 255.0

        # create a Texture object
        texture = Texture(texture_data)

        # return the texture
        return texture
