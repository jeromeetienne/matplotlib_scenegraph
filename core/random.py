import numpy as np


class Random:
    __np_random_generator = np.random.default_rng()

    @staticmethod
    def set_random_seed(seed: int):
        """Set the random seed for numpy for reproducibility.

        Args:
            seed (int): The seed value to set.
        """
        Random.__np_random_generator = np.random.default_rng(seed)

    @staticmethod
    def random() -> float:
        """Generate a random float between 0 and 1.

        Returns:
            float: A random float between 0 and 1.
        """
        return Random.__np_random_generator.random()

    @staticmethod
    def random_uuid() -> str:
        """Generate a random UUID string.

        Returns:
            str: A randomly generated UUID string.
        """

        # Generate 16 random bytes
        bytes = [Random.__np_random_generator.integers(0, 255) for _ in range(16)]
        # Set version to 4 (random)
        bytes[6] = (bytes[6] & 0x0F) | 0x40
        # Set variant to RFC 4122
        bytes[8] = (bytes[8] & 0x3F) | 0x80
        # Format as a UUID string
        uuid_str = (
            '{:02x}{:02x}{:02x}{:02x}-'
            '{:02x}{:02x}-'
            '{:02x}{:02x}-'
            '{:02x}{:02x}-'
            '{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}'
        ).format(*bytes)

        return uuid_str