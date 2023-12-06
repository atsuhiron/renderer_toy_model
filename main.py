import json

import numpy as np

import geom
import world as wrd
import algorithm


if __name__ == "__main__":
    import importlib
    importlib.reload(algorithm)

    with open("samples/simple_world.json", "r") as f:
        wd = json.load(f)
    sw = wrd.World.from_dict(wd)

    init_part = geom.Particle(
        pos=sw.camera.pos,
        vec=algorithm.normalize(np.array([-1.0, 3.0, 0.0], dtype=np.float32))
    )

    ret = algorithm.find_collision_surface(init_part, sw.surfaces)
