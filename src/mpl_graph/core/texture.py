# pip imports
import numpy as np
import matplotlib.image


class Texture:
    __slots__ = ("data",)

    def __init__(self, data: np.ndarray) -> None:
        assert data.ndim == 3 and data.shape[2] in [3, 4], "image should be of shape [H, W, 3] or [H, W, 4]"
        assert data.dtype == np.uint8, "image data type should be uint8"

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

    @staticmethod
    def from_file(file_path: str) -> "Texture":
        """
        Load a texture image from file.

        Arguments:
            file_path (str): Path to the image file.
        Returns:
            np.ndarray: Loaded image as a numpy array.
        """
        texture_data = matplotlib.image.imread(file_path)
        if texture_data.dtype != np.uint8:
            # convert to uint8
            texture_data = (texture_data * 255).astype(np.uint8)

        texture = Texture(texture_data)

        return texture
