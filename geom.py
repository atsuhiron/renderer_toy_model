from __future__ import annotations
from typing import Iterable
from typing import Any
import uuid

import matplotlib.pyplot as plt
import numpy as np

import chromatic
import base_geom
import algorithm


class Particle(base_geom.BaseParticle):
    def __init__(self, pos: np.ndarray, vec: np.ndarray,
                 parent_ids: list[str] = None, intensity: float = 1.0, light: chromatic.CLight = None,
                 last_collided_surface_id: str = None, is_terminated: bool = False):
        self._uuid = uuid.uuid1()
        self._pos = pos
        self._vec = vec
        self._intensity = intensity
        self._terminated = is_terminated
        self._last_collided_surface_id = last_collided_surface_id

        if parent_ids is None:
            self._parents_ids = []
            self._generation = 1
        else:
            self._parents_ids = parent_ids
            self._generation = 1 + len(parent_ids)

        if light is None:
            self._light = chromatic.CLight(np.ones(3))
        else:
            self._light = light

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

    def get_parent_ids(self, contain_self: bool = True) -> list[str]:
        if contain_self:
            return self._parents_ids + [str(self._uuid)]
        return self._parents_ids

    def get_generation(self) -> int:
        return self._generation

    def is_terminated(self) -> bool:
        return self._terminated

    def get_color(self) -> np.ndarray:
        return self._light.to_color().get_array()

    def get_last_collided_surface_id(self) -> str:
        return self._last_collided_surface_id

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

    @staticmethod
    def create_terminated_particle(source: Particle, light_element: np.ndarray) -> Particle:
        return Particle(source.get_pos(),
                        algorithm.normalize(source.get_vec()),
                        source.get_parent_ids(contain_self=False),
                        source.get_intensity(),
                        chromatic.CLight(light_element),
                        None,  # None で大丈夫なはず
                        True)


class SmoothSurface(base_geom.BaseSurface):
    def get_collision_particle(self, in_part: Particle, num: int, c_param: np.ndarray) -> list[Particle]:
        c_point = self.calc_relative_c_point(c_param) + self.get_origin()
        out_vec = algorithm.normalize(algorithm.calc_main_out_vec(self, in_part))
        return [Particle(c_point, out_vec, in_part.get_parent_ids(), in_part.get_intensity(), last_collided_surface_id=self.get_id())]

    @staticmethod
    def get_surface_type() -> str:
        return "smooth"

    @staticmethod
    def from_dict(surface_dict: dict[str, Any]) -> SmoothSurface:
        p1 = np.array(surface_dict["point1"], dtype=np.float32)
        p2 = np.array(surface_dict["point2"], dtype=np.float32)
        p3 = np.array(surface_dict["point3"], dtype=np.float32)
        name = surface_dict.get("name")
        return SmoothSurface(p1, p2, p3, name)


class RoughSurface(base_geom.BaseSurface):
    SAMPLE_COEF = np.array([np.pi * 0.5, 1], dtype=np.float32)

    def __init__(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray,
                 color: chromatic.CColor, name: str = None):
        super().__init__(point1, point2, point3, name)
        self._color = color

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
        rel_c_point = self.calc_relative_c_point(c_param)
        c_point = rel_c_point + self.get_origin()

        # azimuth, from uniform distribution
        phi = np.random.random(num) * 2 * np.pi
        # zenith, from cosine distribution
        theta = self._sample_from_cos_distribution(num)

        normalized_norm = algorithm.normalize(self.get_norm_vec())
        if np.dot(normalized_norm, in_part.get_vec()) > 0:
            normalized_norm *= -1

        c_point_to_origin_vec = -algorithm.normalize(rel_c_point)

        out_particles = []
        child_intensity = in_part.get_intensity() / num
        for ii in range(num):
            zenith_rotation_axial_vec = algorithm.rotate_vector(c_point_to_origin_vec, normalized_norm, float(phi[ii]))
            out_vec = algorithm.normalize(algorithm.rotate_vector(normalized_norm, zenith_rotation_axial_vec, float(theta[ii])))
            out_particles.append(
                Particle(c_point, out_vec, in_part.get_parent_ids(), child_intensity, last_collided_surface_id=self.get_id())
            )

        return out_particles

    @staticmethod
    def get_surface_type() -> str:
        return "rough"

    @staticmethod
    def from_dict(surface_dict: dict[str, Any]) -> RoughSurface:
        p1 = np.array(surface_dict["point1"], dtype=np.float32)
        p2 = np.array(surface_dict["point2"], dtype=np.float32)
        p3 = np.array(surface_dict["point3"], dtype=np.float32)
        color = chromatic.CColor(surface_dict["color"])
        name = surface_dict.get("name")
        return RoughSurface(p1, p2, p3, color, name)


class LightSurface(base_geom.BaseSurface):
    def __init__(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray,
                 light: chromatic.CLight, name: str = None):
        super().__init__(point1, point2, point3, name)
        self._light = light

    def get_collision_particle(self, in_part: Particle, num: int, c_param: np.ndarray) -> list[Particle]:
        return [Particle.create_terminated_particle(in_part, self._light.get_array())]

    @staticmethod
    def get_surface_type() -> str:
        return "light"

    @staticmethod
    def from_dict(surface_dict: dict[str, Any]) -> LightSurface:
        p1 = np.array(surface_dict["point1"], dtype=np.float32)
        p2 = np.array(surface_dict["point2"], dtype=np.float32)
        p3 = np.array(surface_dict["point3"], dtype=np.float32)
        light = chromatic.CLight(surface_dict["light"])
        name = surface_dict.get("name")
        return LightSurface(p1, p2, p3, light, name)


if __name__ == "__main__":
    import importlib
    import viewer

    importlib.reload(viewer)
    _y = 12
    surface = LightSurface(
        np.array([_y, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        # np.array([_y, 1, 0]),
        light=chromatic.CLight(np.array([42, 53, 9], dtype=np.float32)/255)
    )

    np.random.seed(24)
    particles = [Particle(np.zeros(3), np.random.random(3)) for _ in range(20)]
    child_particles = []
    cp_arr = []
    for particle in particles:
        collision_param = algorithm.calc_collision_param(surface, particle)
        cp_arr.append(collision_param)
        if algorithm.do_collision(collision_param, surface.get_basis_norm()):
            child_particles += surface.get_collision_particle(particle, 4, collision_param)
    cp_arr = np.array(cp_arr)
    colors = ["red"] * len(particles) + ["green"] * len(child_particles)
    vectors = np.array([p.get_vec() for p in particles + child_particles])
    locations = np.array([p.get_pos() for p in particles + child_particles])
    viewer.vector(vectors, locations, colors, [surface.get_points()])

    org = surface.get_origin()[::2]
    basis = surface.get_basis()[:, ::2]
    plt.plot(surface.get_points()[:, 0], surface.get_points()[:, 2])
    for cp in cp_arr:
        _rel_c_point = np.sum(basis * np.array([[cp[0]], [cp[1]]]), axis=0)
        _c_point = _rel_c_point + org
        plt.plot([_c_point[0]], [_c_point[1]], "o", label=f"{cp[0]:.3f}")
    plt.legend()
    plt.show()

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
