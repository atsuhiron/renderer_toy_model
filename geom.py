from typing import Iterable

import abc
import uuid

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


class Particle:
    def __init__(self, pos: np.ndarray, vec: np.ndarray,
                 parent_ids: list[str] = None, intensity: float = 1.0):
        self._uuid = uuid.uuid1()
        self._pos = pos
        self._vec = vec
        self._intensity = intensity
        self._terminated = False

        if parent_ids is None:
            self._parents_ids = []
            self._generation = 1
        else:
            self._parents_ids = parent_ids
            self._generation = 1 + len(parent_ids)

    def __str__(self) -> str:
        p = self._fmt_list(self._pos)
        v = self._fmt_list(self._vec)
        i = self._fmt(self._intensity)
        g = self._fmt(self._generation)
        return f"Particle(pos: {p}, vec: {v}, itst: {i}, gen: {g})"

    def get_pos(self) -> np.ndarray:
        return self._pos

    def get_vec(self) -> np.ndarray:
        return self._vec

    def get_intensity(self) -> float:
        return self._intensity

    def get_id(self) -> str:
        return str(self._uuid)

    def get_ids_with_self(self) -> list[str]:
        return self._parents_ids + [self.get_id()]

    def get_generation(self) -> int:
        return self._generation

    def set_terminated(self, is_terminated: bool):
        self._terminated = is_terminated

    @staticmethod
    def _fmt(val: int | float) -> str:
        if type(val) is int:
            return str(val)

        val_str = f"{val:.3f}"
        if val >= 0:
            return " " + val_str
        return val_str

    @staticmethod
    def _fmt_list(vals: Iterable) -> str:
        str_list = [Particle._fmt(v) for v in vals]
        return "[" + ", ".join(str_list) + "]"


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

    def get_points(self) -> np.ndarray:
        return self._points

    @abc.abstractmethod
    def get_collision_particle(self, in_part: Particle, num: int, c_param: np.ndarray) -> list[Particle]:
        pass


class SmoothSurface(Surface):
    def get_collision_particle(self, in_part: Particle, num: int, c_param: np.ndarray) -> list[Particle]:
        e1, e2, c = c_param
        c_point = np.sum(self.get_basis() * np.array([[e1], [e2]]), axis=0) + self.get_origin()
        out_vec = calc_main_out_vec(self, in_part)
        return [Particle(c_point, out_vec, in_part.get_ids_with_self(), in_part.get_intensity())]


class RoughSurface(Surface):
    SAMPLE_COEF = np.array([np.pi * 0.5, 1], dtype=np.float32)

    @staticmethod
    def _sample_from_cos_distribution(size: int) -> np.ndarray:
        # TODO: low efficiency
        samples = np.zeros(size)

        for index in range(size):
            while True:
                angle, value = np.random.random(2) * RoughSurface.SAMPLE_COEF
                if angle > value:
                    samples[index] = angle
                    break
        return samples

    def get_collision_particle(self, in_part: Particle, num: int, c_param: np.ndarray) -> list[Particle]:
        e1, e2, c = c_param
        rel_c_point = np.sum(self.get_basis() * np.array([[e1], [e2]]), axis=0)
        c_point = rel_c_point + self.get_origin()

        # azimuth, from uniform distribution
        phi = np.random.random(num) * 2 * np.pi
        # zenith, from cosine distribution
        theta = self._sample_from_cos_distribution(num)

        normalized_norm = -normalize(self.get_norm_vec())
        c_point_to_origin_vec = -normalize(rel_c_point)

        out_particles = []
        child_intensity = in_part.get_intensity() / num
        for ii in range(num):
            zenith_rotation_axial_vec = rotate_vector(c_point_to_origin_vec, normalized_norm, float(phi[ii]))
            out_vec = rotate_vector(normalized_norm, zenith_rotation_axial_vec, float(theta[ii]))
            out_particles.append(
                Particle(c_point, out_vec, in_part.get_ids_with_self(), child_intensity)
            )

        return out_particles


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


# def calc_collision_angle(suf: Surface, part: Particle) -> float:
#     norm = suf.get_norm_vec()
#     vec = part.get_vec()
#     cos_theta = np.dot(norm, vec) / np.linalg.norm(norm) / np.linalg.norm(vec)
#     return np.arccos(np.abs(cos_theta))


def rotate_vector(vec: np.ndarray, normalized_axial: np.ndarray, radian: float) -> np.ndarray:
    # Reference: http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/tech0007.html
    r = np.array([
        [0, -normalized_axial[2], normalized_axial[1]],
        [normalized_axial[2], 0, -normalized_axial[0]],
        [-normalized_axial[1], normalized_axial[0], 0]
    ])
    m = np.eye(3) + np.sin(radian) * r + (1 - np.cos(radian)) * np.dot(r, r)
    qm = np.eye(4)
    qm[0:3, 0:3] = m

    qv = np.ones(4)
    qv[0:3] = vec
    return np.dot(qm, qv)[0:3]


def calc_main_out_vec(suf: Surface, part: Particle) -> np.ndarray:
    norm = normalize(suf.get_norm_vec())

    in_vec = -part.get_vec()  # inverse direction
    return rotate_vector(in_vec, norm, np.pi)


if __name__ == "__main__":
    import importlib
    import plot

    importlib.reload(plot)

    surface = RoughSurface(
        1111,
        np.array([12, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),

    )

    particles = [Particle(np.zeros(3), np.random.random(3)) for _ in range(20)]
    child_particles = []
    for particle in particles:
        collision_param = calc_collision_param(surface, particle)
        if do_collision(collision_param):
            child_particles += surface.get_collision_particle(particle, 4, collision_param)

    colors = ["red"] * len(particles) + ["green"] * len(child_particles)
    vectors = np.array([p.get_vec() for p in particles + child_particles])
    locations = np.array([p.get_pos() for p in particles + child_particles])
    plot.vector(vectors, locations, colors, [surface.get_points()])

    # n = normalize(np.ones(3))
    # v = normalize(np.random.random(3))
    # vectors = []
    # for ang in np.linspace(0, np.pi, 20):
    #     rv = rotate_vector(v, n, ang)
    #     vectors.append(rv)
    #
    # colors = ["red"] * len(vectors)
    # colors.append("k")
    #
    # vectors.append(n)
    # vectors = np.array(vectors)
    # locations = np.zeros((len(vectors), 3))
    #
    # plot.vector(vectors, locations, colors)
