import geom
from camera import Camera


class World:
    def __init__(self, surfaces: list[geom.Surface], camera: Camera):
        self.surfaces = surfaces
        self.camera = camera
