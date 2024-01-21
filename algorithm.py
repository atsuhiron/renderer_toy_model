import numpy as np
import numba

import base_geom
from const import NUMBA_OPT


@numba.jit("f4[:](f4[:])", **NUMBA_OPT)
def normalize(vec: np.ndarray) -> np.ndarray:
    sq = vec * vec
    return vec / np.sqrt(sq[0] + sq[1] + sq[2])


@numba.jit(**NUMBA_OPT)
def _gen_a(suf_basis: np.ndarray, part_vec: np.ndarray) -> np.ndarray:
    return np.array([
        [suf_basis[0, 0], suf_basis[1, 0], part_vec[0]],
        [suf_basis[0, 1], suf_basis[1, 1], part_vec[1]],
        [suf_basis[0, 2], suf_basis[1, 2], part_vec[2]],
    ], dtype=np.float32)


def calc_collision_param(suf: base_geom.BaseSurface, part: base_geom.BaseParticle) -> np.ndarray:
    a = _gen_a(suf.get_basis(), -part.get_vec())
    if np.linalg.matrix_rank(a) < 3:
        # particle and surface is parallel (never collide)
        return -np.ones(3, dtype=np.float32)

    b = part.get_pos() - suf.get_origin()
    return np.linalg.solve(a, b)


@numba.jit(**NUMBA_OPT)
def do_collision(c_param: np.ndarray, basis_norm: np.ndarray) -> bool:
    e1, e2, c = c_param
    e1 /= basis_norm[0]
    e2 /= basis_norm[1]

    if e1 < 0 or e2 < 0:
        return False

    if (e1 + e2) > 0.5:
        return False

    return True


@numba.jit(**NUMBA_OPT)
def _dot44_4(mat44: np.ndarray, vec4: np.ndarray) -> np.ndarray:
    return np.array([
        np.sum(mat44[0] * vec4),
        np.sum(mat44[1] * vec4),
        np.sum(mat44[2] * vec4),
    ], dtype=np.float32)


@numba.jit(**NUMBA_OPT)
def _dot33_33(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return np.array([
        [np.sum(m1[0] * m2[:, 0]), np.sum(m1[0] * m2[:, 1]), np.sum(m1[0] * m2[:, 2])],
        [np.sum(m1[1] * m2[:, 0]), np.sum(m1[1] * m2[:, 1]), np.sum(m1[1] * m2[:, 2])],
        [np.sum(m1[2] * m2[:, 0]), np.sum(m1[2] * m2[:, 1]), np.sum(m1[2] * m2[:, 2])]
    ], dtype=np.float32)


@numba.jit(**NUMBA_OPT)
def rotate_vector(vec: np.ndarray, normalized_axial: np.ndarray, radian: float) -> np.ndarray:
    # Reference: http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/tech0007.html
    r = np.array([
        [0, -normalized_axial[2], normalized_axial[1]],
        [normalized_axial[2], 0, -normalized_axial[0]],
        [-normalized_axial[1], normalized_axial[0], 0]
    ], dtype=np.float32)

    sin = np.float32(np.sin(radian))
    cos = np.float32(np.cos(radian))
    m = np.eye(3, dtype=np.float32) + sin * r + (1 - cos) * _dot33_33(r, r)
    qm = np.eye(4, dtype=np.float32)
    qm[0:3, 0:3] = m
    qv = np.ones(4, dtype=np.float32)
    qv[0:3] = vec
    return _dot44_4(qm, qv)


def calc_main_out_vec(suf: base_geom.BaseSurface, part: base_geom.BaseParticle) -> np.ndarray:
    norm = normalize(suf.get_norm_vec())

    in_vec = -part.get_vec()  # inverse direction
    if np.dot(norm, in_vec) > 0:
        norm *= -1
    return rotate_vector(in_vec, norm, np.pi)


def find_collision_surface(part: base_geom.BaseParticle, surfaces: list[base_geom.BaseSurface]) -> tuple[np.ndarray, base_geom.BaseSurface | None]:
    # TODO: need pre filtering (ex. Exclude surface in completely different direction)
    collisions = []
    for suf in surfaces:
        if part.get_last_collided_surface_id() == suf.get_id():
            continue

        c_param = calc_collision_param(suf, part)
        if do_collision(c_param, suf.get_basis_norm()):
            collisions.append((c_param, suf))

    if len(collisions) == 0:
        return -np.ones(3), None
    return min(collisions, key=lambda col: col[0][2])
