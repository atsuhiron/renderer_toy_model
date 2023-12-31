from __future__ import annotations
import abc
import uuid

import numpy as np

import chromatic


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
    def get_light(self) -> chromatic.CLight:
        pass

    @abc.abstractmethod
    def get_last_collided_surface_id(self) -> str:
        pass

    @abc.abstractmethod
    def get_uuid(self) -> uuid.UUID:
        pass

    @staticmethod
    @abc.abstractmethod
    def create_inverse_traced_particle(source: BaseParticle, light: chromatic.CLight, itst: float) -> BaseParticle:
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

        self._basis = None
        self._basis_norm = None
        self._norm_vec = None

    def get_name(self) -> str:
        return self._name

    def get_basis(self) -> np.ndarray:
        if self._basis is None:
            self._basis = np.array([self._points[1] - self._points[0], self._points[2] - self._points[0]])
        return self._basis

    def get_basis_norm(self) -> np.ndarray:
        if self._basis_norm is None:
            self._basis_norm = np.linalg.norm(self.get_basis(), axis=1)
        return self._basis_norm

    def get_origin(self) -> np.ndarray:
        return self._points[0]

    def get_norm_vec(self) -> np.ndarray:
        if self._norm_vec is None:
            e1, e2 = self.get_basis()
            self._norm_vec = np.cross(e1, e2)
        return self._norm_vec

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
