from __future__ import annotations
from typing import Any

import geom
from camera import Camera


class World:
    def __init__(self, surfaces: list[geom.Surface], camera: Camera):
        self.surfaces = surfaces
        self.camera = camera

    @staticmethod
    def from_dict(world_dict: dict[str, Any]) -> World:
        _camera = world_dict["camera"]
        surfaces = []
        for suf_dict in world_dict["surfaces"]:
            suf_type = suf_dict["surface_type"]
            if suf_type == "smooth":
                surfaces.append(geom.SmoothSurface.from_dict(suf_dict))
            elif suf_type == "rough":
                surfaces.append(geom.RoughSurface.from_dict(suf_dict))
            elif suf_type == "light":
                surfaces.append(geom.LightSurface.from_dict(suf_dict))
            else:
                assert False, f"Unknown surface type {suf_type}"

        return World(surfaces, _camera)


if __name__ == "__main__":
    point = [0.2, 0.4, 0.7]
    wd = {
        "camera": {
            "pos": point,
            "vec": point,
            "focal_length": 1.2,
            "fov_v": 1.57,
            "fov_h": 1.57,
            "pixel_v": 100,
            "pixel_h": 100,
        },
        "surfaces": [
            {
                "surface_type": "smooth",
                "point1": point,
                "point2": point,
                "point3": point,
            },
            {
                "surface_type": "rough",
                "point1": point,
                "point2": point,
                "point3": point,
                "color": [0.1, 0.5, 1.0]
            },
            {
                "surface_type": "light",
                "point1": point,
                "point2": point,
                "point3": point,
                "light": [0.9, 0.5, 0.0]
            }
        ]
    }

    world = World.from_dict(wd)
