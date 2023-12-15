from typing import Generator
import argparse
import json
import time
import os
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tqdm

import base_geom as bg
import chromatic
import geom
import world as wrd
import algorithm
import rendering_config
import viewer


def render(world: wrd.World, config: rendering_config.RenderingConfig, para_num: int = None) -> list[bg.BaseParticle]:
    cam = world.camera

    start = time.time()
    generations = [cam.create_pixel_particles()]
    if para_num is None:
        # Non parallel
        for g in range(1, config.max_generation + 1):
            children = trace_particles(generations[g - 1], world.surfaces, config)
            generations.append(children)

        inverse_traced_child = generations[-1]
        for g in range(1, config.max_generation + 1)[::-1]:
            inverse_traced_child = inverse_trace(inverse_traced_child, generations[g - 1])
    else:
        # Parallel
        with Pool(para_num) as pool:
            for g in range(1, config.max_generation + 1):
                children = trace_particles_para(generations[g - 1], world.surfaces, config, pool)
                generations.append(children)

        inverse_traced_child = generations[-1]
        for g in range(1, config.max_generation + 1)[::-1]:
            inverse_traced_child = inverse_trace(inverse_traced_child, generations[g - 1])

    end = time.time()
    print(f"Rendering time: {end - start:.4f} s")
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
    return col_suf.get_collision_particle(part, config.rough_surface_child_num, c_param)


def trace_particle_wrap(psc: tuple[bg.BaseParticle, list[bg.BaseSurface], rendering_config.RenderingConfig]) -> list[bg.BaseParticle]:
    return trace_particle(*psc)


def trace_particles(particles: list[bg.BaseParticle],
                    surfaces: list[bg.BaseSurface],
                    config: rendering_config.RenderingConfig) -> list[bg.BaseParticle]:
    children = []
    for part in tqdm.tqdm(particles):
        children += trace_particle(part, surfaces, config)
    return children


def _gen_arg(particles: list[bg.BaseParticle],
             surfaces: list[bg.BaseSurface],
             config: rendering_config.RenderingConfig) -> Generator[tuple[bg.BaseParticle, list[bg.BaseSurface], rendering_config.RenderingConfig], None, None]:
    for part in particles:
        yield part, surfaces, config


def _gen_arg2(particles: list[bg.BaseParticle],
              surfaces: list[bg.BaseSurface],
              config: rendering_config.RenderingConfig,
              p_bar: tqdm.tqdm) -> Generator[tuple[bg.BaseParticle, list[bg.BaseSurface], rendering_config.RenderingConfig], None, None]:
    for part in particles:
        yield part, surfaces, config
        p_bar.update(1)


def trace_particles_para(particles: list[bg.BaseParticle],
                         surfaces: list[bg.BaseSurface],
                         config: rendering_config.RenderingConfig,
                         pool: Pool) -> list[bg.BaseParticle]:
    trace_res = []
    with tqdm.tqdm(total=len(particles)) as p_bar:
        for part in pool.imap_unordered(trace_particle_wrap, _gen_arg(particles, surfaces, config), chunksize=12):
            trace_res.append(part)
            p_bar.update(1)

    children = []
    for res in trace_res:
        children += res
    return children


def trace_particles_para2(particles: list[bg.BaseParticle],
                          surfaces: list[bg.BaseSurface],
                          config: rendering_config.RenderingConfig,
                          pool: ProcessPoolExecutor) -> list[bg.BaseParticle]:
    with tqdm.tqdm(total=len(particles)) as p_bar:
        trace_res = pool.map(trace_particle_wrap, _gen_arg2(particles, surfaces, config, p_bar), chunksize=12)

    children = []
    for res in trace_res:
        children += res
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


def _parse_parallel(value: str) -> int | None:
    if value.isdigit():
        d = int(value)
        if d > 1:
            return d
        return None
    if isinstance(value, str):
        if value.lower() == "auto":
            return os.cpu_count()
        if value.lower() == "none":
            return None
    raise ValueError(f"Unknown parallel value: {value}")


if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(description='Rendering CG')
    parser.add_argument("world_json_path", help="The json file path describing 3d model.")
    parser.add_argument("-g", "--max-gen", type=int, default=3, help="Max child particle generation.")
    parser.add_argument("-c", "--child-num", type=int, default=6,
                        help="The number of child created by a parent particle.")
    parser.add_argument("-p", "--parallel", default="auto", help="The number of process. If the number is less than 2 or 'none' is specified, all procedure is done in the main thread.")
    args = parser.parse_args()

    with open(args.world_json_path, "r") as f:
        wd = json.load(f)
    r_config = rendering_config.RenderingConfig(args.max_gen, args.child_num)
    sw = wrd.World.from_dict(wd)
    inverse_traced = render(sw, r_config, _parse_parallel(args.parallel))

    viewer.show(inverse_traced, sw.camera)
