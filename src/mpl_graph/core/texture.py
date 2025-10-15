# pip imports
import numpy as np
import matplotlib.image


class Texture:
    __slots__ = ("data",)

    def __init__(self, data: np.ndarray | None = None) -> None:
        """
        float texture image data of shape [H, W, 3] or [H, W, 4] in range [0, 1]
        """

        self.data: np.ndarray = data if data is not None else np.array([], dtype=np.float32).reshape((0, 0, 3))
        """float texture image data of shape [H, W, 3] or [H, W, 4] in range [0, 1]"""

        assert self.data.ndim == 3 and self.data.shape[2] in [3, 4], f"image should be of shape [H, W, 3] or [H, W, 4], got {self.data.shape}"
        assert self.data.dtype in [np.float32, np.float64], f"image should be of type float32 or float64, got {self.data.dtype}"

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
        return self.width() / self.height() if self.height() != 0 else 1.0

    def has_alpha(self) -> bool:
        """Return True if the texture has an alpha channel."""
        return self.data.shape[2] == 4

    def strip_alpha(self) -> "Texture":
        """Strip the texture of the alpha channel in place."""

        if not self.has_alpha():
            return self

        self.data = self.data[:, :, :3]

        return self

    def ensure_no_alpha(self) -> "Texture":
        """Ensure the texture has no alpha channel, stripping it if necessary."""
        return self.strip_alpha() if self.has_alpha() else self

    @staticmethod
    def from_file(file_path: str) -> "Texture":
        """
        Load a texture image from file.
        """

        # read image using matplotlib
        texture_data = matplotlib.image.imread(file_path)

        # convert uint8 to float32 in range [0, 1] if needed
        if texture_data.dtype == np.uint8:
            # convert to float32 in range [0, 1]
            texture_data = texture_data.astype(np.float32) / 255.0

        # create a Texture object
        texture = Texture(texture_data)

        # return the texture
        return texture
