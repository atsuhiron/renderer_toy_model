import json

import numpy as np
import tqdm

import base_geom
import geom
import world as wrd
import algorithm
import rendering_config
import viewer


def render(world: wrd.World, config: rendering_config.RenderingConfig):
    pass


def trace_particle(part: base_geom.BaseParticle,
                   surfaces: list[base_geom.BaseSurface],
                   config: rendering_config.RenderingConfig) -> list[base_geom.BaseParticle]:
    if part.get_generation() > config.max_generation:
        return []
    if part.is_terminated():
        return []

    c_param, col_suf = algorithm.find_collision_surface(part, surfaces)

    if col_suf is None:
        return []
    return col_suf.get_collision_particle(part, r_config.rough_surface_child_num, c_param)


def trace_particles(particles: list[base_geom.BaseParticle],
                    surfaces: list[base_geom.BaseSurface],
                    config: rendering_config.RenderingConfig) -> list[base_geom.BaseParticle]:
    children = []
    for part in tqdm.tqdm(particles):
        children += trace_particle(part, surfaces, config)
    return children


if __name__ == "__main__":
    import importlib
    importlib.reload(algorithm)

    with open("samples/simple_world.json", "r") as f:
        wd = json.load(f)
    r_config = rendering_config.RenderingConfig(2, 4)

    sw = wrd.World.from_dict(wd)
    init_part = geom.Particle(
        pos=sw.camera.pos,
        vec=algorithm.normalize(np.array([-1.0, 3.0, 0.0], dtype=np.float32))
    )

    child_1g = trace_particles([init_part], sw.surfaces, r_config)
    child_2g = trace_particles(child_1g, sw.surfaces, r_config)
    child_3g = trace_particles(child_2g, sw.surfaces, r_config)

    # plot
    colors = ["red"] * 1 + ["green"] * len(child_1g) + ["blue"] * len(child_2g)
    vectors = np.array([p.get_vec() for p in [init_part] + child_1g + child_2g])
    locations = np.array([p.get_pos() for p in [init_part] + child_1g + child_2g])
    viewer.vector(vectors, locations, colors, [suf.get_points() for suf in sw.surfaces])

    # 後処理で LightSurface に衝突した Particle とその前の世代の Particle が重複するのに対処する必要がある
