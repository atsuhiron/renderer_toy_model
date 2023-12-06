from __future__ import annotations
from typing import Any

import base_geom
import geom
from camera import Camera


class World:
    def __init__(self, surfaces: list[base_geom.BaseSurface], camera: Camera):
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
    import json

    with open("samples/simple_world.json", "r") as f:
        wd = json.load(f)
    world = World.from_dict(wd)
