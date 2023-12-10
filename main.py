import json

import numpy as np
import tqdm

import base_geom as bg
import chromatic
import geom
import world as wrd
import algorithm
import rendering_config
import viewer


def render(world: wrd.World, config: rendering_config.RenderingConfig):
    pass


def trace_particle(part: bg.BaseParticle,
                   surfaces: list[bg.BaseSurface],
                   config: rendering_config.RenderingConfig) -> list[bg.BaseParticle]:
    if part.get_generation() > config.max_generation:
        return []
    if part.is_terminated():
        return []

    c_param, col_suf = algorithm.find_collision_surface(part, surfaces)

    if col_suf is None:
        return []
    return col_suf.get_collision_particle(part, r_config.rough_surface_child_num, c_param)


def trace_particles(particles: list[bg.BaseParticle],
                    surfaces: list[bg.BaseSurface],
                    config: rendering_config.RenderingConfig) -> list[bg.BaseParticle]:
    children = []
    for part in tqdm.tqdm(particles):
        children += trace_particle(part, surfaces, config)
    return children


def inverse_trace_child(children: list[bg.BaseParticle], parent: bg.BaseParticle) -> bg.BaseParticle:
    light = np.array([c.get_light().get_array() for c in children])
    itst = np.array([c.get_intensity() for c in children])
    synthesis_light, synthesis_itst = chromatic.add_light(light, itst)
    new_light = chromatic.CLight(synthesis_light)
    return geom.Particle.create_inverse_traced_particle(parent, new_light, synthesis_itst)


def inverse_trace(children: list[bg.BaseParticle], parents: list[bg.BaseParticle]) -> list[bg.BaseParticle]:
    parent_ids = [p.get_parent_ids()[-1] for p in parents]
    family_tree = {pid: [] for pid in parent_ids}

    for c in children:
        family_tree[c.get_parent_ids(False)[-1]].append(c)

    inverse_traced_parents = []
    for index, pid in enumerate(parent_ids):
        family = family_tree[pid]
        if len(family) == 0:
            itp = parents[index]
        else:
            itp = inverse_trace_child(family, parents[index])
        inverse_traced_parents.append(itp)

    return inverse_traced_parents


if __name__ == "__main__":
    import importlib
    importlib.reload(algorithm)
    np.random.seed(8)

    with open("samples/simple_world.json", "r") as f:
        wd = json.load(f)
    r_config = rendering_config.RenderingConfig(3, 4)

    sw = wrd.World.from_dict(wd)
    part_0g = [geom.Particle(
        pos=sw.camera.pos,
        vec=algorithm.normalize(np.array([-1.0, 3.0, 0.0], dtype=np.float32))
    )]

    child_1g = trace_particles(part_0g, sw.surfaces, r_config)
    child_2g = trace_particles(child_1g, sw.surfaces, r_config)
    child_3g = trace_particles(child_2g, sw.surfaces, r_config)

    inverse_2g = inverse_trace(child_3g, child_2g)
    inverse_1g = inverse_trace(inverse_2g, child_1g)
    inverse_0g = inverse_trace(inverse_1g, part_0g)

    # plot
    # colors = ["red"] * 1 + ["green"] * len(child_1g) + ["blue"] * len(child_2g)
    # vectors = np.array([p.get_vec() for p in [init_part] + child_1g + child_2g])
    # locations = np.array([p.get_pos() for p in [init_part] + child_1g + child_2g])
    # viewer.vector(vectors, locations, colors, [suf.get_points() for suf in sw.surfaces])
