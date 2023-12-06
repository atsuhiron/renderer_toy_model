from __future__ import annotations
from typing import Any

import numpy as np


class Camera:
    def __init__(self, pos: np.ndarray, vec: np.ndarray, focal_length: float,
                 vertical_fov: float, horizontal_fov: float, vertical_pixel: int, horizontal_pixel: int):
        self.pos = pos
        self.vec = vec
        self.focal = focal_length
        self.fov_v = vertical_fov
        self.fov_h = horizontal_fov
        self.pixel_v = vertical_pixel
        self.pixel_h = horizontal_pixel

    @staticmethod
    def from_dict(camera_dict: dict[str, Any]) -> Camera:
        pos = np.array(camera_dict["pos"], dtype=np.float32)
        vec = np.array(camera_dict["vec"], dtype=np.float32)
        focal_length = float(camera_dict["focal_length"])
        fov_v = camera_dict["fov_v"]
        fov_h = camera_dict["fov_h"]
        pixel_v = camera_dict["pixel_v"]
        pixel_h = camera_dict["pixel_h"]
        return Camera(pos, vec, focal_length, fov_v, fov_h, pixel_v, pixel_h)
