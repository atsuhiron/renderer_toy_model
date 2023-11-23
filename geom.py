import abc

import matplotlib.pyplot as plt
import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


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


def rotate_vector(vec: np.ndarray, normalized_axial: np.ndarray, radian: float) -> np.ndarray:
    r = np.array([
        [0, -normalized_axial[2], normalized_axial[1]],
        [normalized_axial[2], 0,  -normalized_axial[0]],
        [-normalized_axial[2], normalized_axial[0], 0]
    ])
    m = np.eye(3) + np.sin(radian) * r + (1 - np.cos(radian)) * np.dot(r, r)
    qm = np.eye(4)
    qm[0:3, 0:3] = m

    qv = np.ones(4)
    qv[0:3] = vec
    print(np.linalg.det(qm))
    return np.dot(qm, qv)[0:3]


def calc_main_out_vec(suf: Surface, part: Particle, c_param: np.ndarray) -> np.ndarray:
    # Reference: http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/tech0007.html
    e1, e2, c = c_param
    c_point = np.sum(suf.get_basis() * np.array([[e1], [e2]]), axis=0) + suf.get_origin()
    norm = normalize(suf.get_norm_vec())


if __name__ == "__main__":
    import importlib
    import plot
    importlib.reload(plot)

    # surface = SmoothSurface(
    #     1111,
    #     np.array([0, 1, 0]),
    #     np.array([0, 1, 1]),
    #     np.array([100, 1, 0]),
    # )
    #
    # particles = []
    # angles = []
    # while len(particles) < 20:
    #     particle = Particle(np.zeros(3), np.random.random(3))
    #
    #     collision_param = calc_collision_param(surface, particle)
    #     if do_collision(collision_param):
    #         ang = calc_collision_angle(surface, particle)
    #         particles.append(particle)
    #         angles.append(ang)
    #
    #         ret = calc_main_out_vec(surface, particle, collision_param)

    n = normalize(np.ones(3))
    v = normalize(np.random.random(3))
    vectors = []
    for ang in np.linspace(0, np.pi, 20):
        rv = rotate_vector(v, n, ang)
        vectors.append(rv)

    colors = ["red"] * len(vectors)
    colors.append("k")

    vectors.append(n)
    vectors = np.array(vectors)
    locations = np.zeros((len(vectors), 3))

    plot.vector(vectors, locations, colors)
