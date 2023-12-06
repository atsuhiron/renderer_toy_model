import abc
import uuid

import numpy as np


class BaseParticle(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_pos(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_vec(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_intensity(self) -> float:
        pass

    @abc.abstractmethod
    def get_parent_ids(self, contain_self: bool = True) -> list[str]:
        pass

    @abc.abstractmethod
    def get_generation(self) -> int:
        pass

    @abc.abstractmethod
    def is_terminated(self) -> bool:
        pass

    @abc.abstractmethod
    def get_color(self) -> np.ndarray:
        pass


class BaseSurface(metaclass=abc.ABCMeta):
    def __init__(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray, name: str = None):
        self._points = np.array([point1, point2, point3], dtype=np.float32)
        self._uuid = uuid.uuid1()
        self._surface_type = self.get_surface_type()
        if name is None:
            self._name = ""
        else:
            self._name = name

    def get_basis(self) -> np.ndarray:
        return np.array([self._points[1] - self._points[0], self._points[2] - self._points[0]])

    def get_basis_norm(self) -> np.ndarray:
        return np.linalg.norm(self.get_basis(), axis=1)

    def get_origin(self) -> np.ndarray:
        return self._points[0]

    def get_norm_vec(self) -> np.ndarray:
        e1, e2 = self.get_basis()
        return np.cross(e1, e2)

    def get_id(self) -> str:
        return str(self._uuid)

    def get_points(self) -> np.ndarray:
        return self._points

    def calc_relative_c_point(self, c_param: np.ndarray) -> np.ndarray:
        e1, e2, c = c_param
        return np.sum(self.get_basis() * np.array([[e1], [e2]]), axis=0)

    @staticmethod
    @abc.abstractmethod
    def get_surface_type() -> str:
        pass

    @abc.abstractmethod
    def get_collision_particle(self, in_part: BaseParticle, num: int, c_param: np.ndarray) -> list[BaseParticle]:
        pass
