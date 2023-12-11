import argparse
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


def render(world: wrd.World, config: rendering_config.RenderingConfig) -> list[bg.BaseParticle]:
    cam = world.camera

    generations = [cam.create_pixel_particles()]
    for g in range(1, config.max_generation + 1):
        children = trace_particles(generations[g - 1], world.surfaces, config)
        generations.append(children)

    inverse_traced_child = generations[-1]
    for g in range(1, config.max_generation + 1)[::-1]:
        inverse_traced_child = inverse_trace(inverse_traced_child, generations[g - 1])
    return inverse_traced_child


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
    parser = argparse.ArgumentParser(description='Rendering CG')
    parser.add_argument("world_json_path", help="The json file path describing 3d model.")
    parser.add_argument("-g", "--max-gen", type=int, default=3, help="Max child particle generation.")
    parser.add_argument("-c", "--child-num", type=int, default=6, help="The number of child created by a parent particle.")
    args = parser.parse_args()

    with open(args.world_json_path, "r") as f:
        wd = json.load(f)
    r_config = rendering_config.RenderingConfig(args.max_gen, args.child_num)
    sw = wrd.World.from_dict(wd)
    inverse_traced = render(sw, r_config)

    viewer.show(inverse_traced, sw.camera)
