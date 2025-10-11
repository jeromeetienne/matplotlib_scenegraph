from abc import ABC, abstractmethod
import numpy as np

from core.object_3d import Object3D


class CameraBase(ABC):
    @abstractmethod
    def get_projection_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_view_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_world_position(self) -> np.ndarray:
        pass
