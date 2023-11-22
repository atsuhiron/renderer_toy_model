import abc

import matplotlib.pyplot as plt
import numpy as np


class Particle:
    def __init__(self, pos: np.ndarray, vec: np.ndarray, intensity: float = 1.0):
        self._pos = pos
        self._vec = vec
        self._intensity = intensity

    def get_pos(self) -> np.ndarray:
        return self._pos

    def get_vec(self) -> np.ndarray:
        return self._vec

    def get_intensity(self) -> float:
        return self._intensity


class Surface(metaclass=abc.ABCMeta):
    def __init__(self, index: int, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray):
        self._points = np.array([point1, point2, point3], dtype=np.float32)
        self._index = index

    def get_basis(self) -> np.ndarray:
        return np.array([self._points[1] - self._points[0], self._points[2] - self._points[0]])

    def get_origin(self) -> np.ndarray:
        return self._points[0]

    def get_norm_vec(self) -> np.ndarray:
        e1, e2 = self.get_basis()
        return np.cross(e1, e2)

    def get_index(self) -> int:
        return self._index

    @abc.abstractmethod
    def get_collision_particle(self, num: int, collision_angle: float) -> list[Particle]:
        pass


class SmoothSurface(Surface):
    def get_collision_particle(self, num: int, collision_angle: float) -> list[Particle]:
        pass


def calc_collision_param(suf: Surface, part: Particle) -> np.ndarray:
    a = np.transpose(np.r_[suf.get_basis(), -part.get_vec()[np.newaxis, :]])
    b = part.get_pos() - suf.get_origin()
    return np.linalg.solve(a, b)


def do_collision(c_param: np.ndarray) -> bool:
    e1, e2, c = c_param

    if e1 < 0 or e2 < 0:
        return False

    if (e1 + e2) > 0.5:
        return False

    return True


def calc_collision_angle(suf: Surface, part: Particle) -> float:
    norm = suf.get_norm_vec()
    vec = part.get_vec()
    cos_theta = np.dot(norm, vec) / np.linalg.norm(norm) / np.linalg.norm(vec)
    return np.arccos(np.abs(cos_theta))


def calc_main_out_vec(suf: Surface, part: Particle, c_param: np.ndarray) -> np.ndarray:
    e1, e2, c = c_param
    c_point = np.sum(suf.get_basis() * np.array([[e1], [e2]]), axis=0) + suf.get_origin()
    


def plot(parts: list[Particle], angs: list[float]):
    pos = np.array([part.get_pos() for part in parts])
    vec = np.array([part.get_vec() for part in parts])
    plt.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1], angs, units='x', scale=np.linalg.norm(vec))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


if __name__ == "__main__":
    surface = SmoothSurface(
        1111,
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        np.array([100, 1, 0]),
    )

    particles = []
    angles = []
    while len(particles) < 20:
        particle = Particle(np.zeros(3), np.random.random(3))

        collision_param = calc_collision_param(surface, particle)
        if do_collision(collision_param):
            ang = calc_collision_angle(surface, particle)
            particles.append(particle)
            angles.append(ang)

            ret = calc_main_out_vec(surface, particle, collision_param)

    plot(particles, angles)
