from __future__ import annotations
from typing import Any

import numpy as np

import algorithm
import geom


class Camera:
    def __init__(self, pos: np.ndarray, vec: np.ndarray, focal_length: float,
                 vertical_fov: float, horizontal_fov: float, vertical_pixel: int, horizontal_pixel: int, mode: str):
        self.pos = pos
        self.vec = vec
        self.focal = focal_length
        self.fov_v = vertical_fov
        self.fov_h = horizontal_fov
        self.pixel_v = vertical_pixel
        self.pixel_h = horizontal_pixel
        self.camera_mode = mode

    def create_pixel_particles(self) -> list[geom.Particle]:
        if self.camera_mode == "plane":
            init_particle_vec = self._plane()
        elif self.camera_mode == "spherical":
            init_particle_vec = self._spherical()
        else:
            assert False, f"Not supported camera mode: {self.camera_mode}"

        particles = []
        for part_vec in init_particle_vec:
            part = geom.Particle(self.pos, part_vec)
            particles.append(part)
        return particles

    def _spherical(self) -> np.ndarray:
        phi = np.linspace(-self.fov_h / 2, self.fov_h / 2, self.pixel_h)
        theta = np.linspace(-self.fov_v / 2, self.fov_v / 2, self.pixel_v)
        pixel_angle = np.zeros((self.pixel_v, self.pixel_h, 3), dtype=np.float32)
        pixel_angle[:, :, 0], pixel_angle[:, :, 2] = np.meshgrid(phi, theta)
        pixel_angle = pixel_angle.reshape((self.pixel_v * self.pixel_h, 3))
        pixel_vec = self.focal * np.sin(pixel_angle)

        # move focal point to origin
        pixel_vec = pixel_vec + np.array([[0.0, self.focal, 0.0]], dtype=np.float32)

        forward = np.array([0, 1, 0], dtype=np.float32)
        if not self._is_forward_direction():
            angle = np.arccos(np.dot(forward, self.vec) / np.linalg.norm(self.vec))
            axial = algorithm.normalize(np.cross(forward, self.vec))
            for ii in range(self.pixel_v * self.pixel_h):
                pixel_vec[ii] = algorithm.rotate_vector(pixel_vec[ii], axial, angle)

        return pixel_vec

    def _plane(self) -> np.ndarray:
        half_v = self.focal * np.sin(self.fov_v / 2)
        half_h = self.focal * np.sin(self.fov_h / 2)
        pixel_vec = np.zeros((self.pixel_v, self.pixel_h, 3), dtype=np.float32)
        pixel_vec[:, :, 0], pixel_vec[:, :, 2] = np.meshgrid(np.linspace(-half_h, half_h, self.pixel_h), np.linspace(-half_v, half_v, self.pixel_v))
        pixel_vec = pixel_vec.reshape((self.pixel_v * self.pixel_h, 3))

        # move focal point to origin
        pixel_vec = pixel_vec + np.array([[0.0, self.focal, 0.0]], dtype=np.float32)

        forward = np.array([0, 1, 0], dtype=np.float32)
        if not self._is_forward_direction():
            angle = np.arccos(np.dot(forward, self.vec) / np.linalg.norm(self.vec))
            axial = algorithm.normalize(np.cross(forward, self.vec))
            for ii in range(self.pixel_v * self.pixel_h):
                pixel_vec[ii] = algorithm.rotate_vector(pixel_vec[ii], axial, angle)

        return pixel_vec

    def _is_forward_direction(self) -> bool:
        return self.vec[0] == 0 and self.vec[2] == 0 and self.vec[1] > 0

    @staticmethod
    def from_dict(camera_dict: dict[str, Any]) -> Camera:
        pos = np.array(camera_dict["pos"], dtype=np.float32)
        vec = np.array(camera_dict["vec"], dtype=np.float32)
        focal_length = float(camera_dict["focal_length"])
        fov_v = camera_dict["fov_v"]
        fov_h = camera_dict["fov_h"]
        pixel_v = camera_dict["pixel_v"]
        pixel_h = camera_dict["pixel_h"]
        camera_mode = camera_dict["camera_mode"]
        return Camera(pos, vec, focal_length, fov_v, fov_h, pixel_v, pixel_h, camera_mode)


if __name__ == "__main__":
    import json
    with open("samples/simple_world.json", "r") as f:
        wd = json.load(f)

    cam = Camera.from_dict(wd["camera"])
    ret = cam.create_pixel_particles()
