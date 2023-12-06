import numpy as np

import base_geom


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def calc_collision_param(suf: base_geom.BaseSurface, part: base_geom.BaseParticle) -> np.ndarray:
    a = np.transpose(np.r_[suf.get_basis(), -part.get_vec()[np.newaxis, :]])
    if np.linalg.matrix_rank(a) < 3:
        # particle and surface is parallel (never collide)
        return -np.ones(3, dtype=np.float32)

    b = part.get_pos() - suf.get_origin()
    return np.linalg.solve(a, b)


def do_collision(c_param: np.ndarray, basis_norm: np.ndarray) -> bool:
    e1, e2, c = c_param
    e1 /= basis_norm[0]
    e2 /= basis_norm[1]

    if e1 < 0 or e2 < 0:
        return False

    if (e1 + e2) > 0.5:
        return False

    return True


def rotate_vector(vec: np.ndarray, normalized_axial: np.ndarray, radian: float) -> np.ndarray:
    # Reference: http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/tech0007.html
    r = np.array([
        [0, -normalized_axial[2], normalized_axial[1]],
        [normalized_axial[2], 0, -normalized_axial[0]],
        [-normalized_axial[1], normalized_axial[0], 0]
    ])
    m = np.eye(3) + np.sin(radian) * r + (1 - np.cos(radian)) * np.dot(r, r)
    qm = np.eye(4)
    qm[0:3, 0:3] = m

    qv = np.ones(4)
    qv[0:3] = vec
    return np.dot(qm, qv)[0:3]


def calc_main_out_vec(suf: base_geom.BaseSurface, part: base_geom.BaseParticle) -> np.ndarray:
    norm = normalize(suf.get_norm_vec())

    in_vec = -part.get_vec()  # inverse direction
    return rotate_vector(in_vec, norm, np.pi)


def find_collision_surface(part: base_geom.BaseParticle, surfaces: list[base_geom.BaseSurface]) -> tuple[float, base_geom.BaseSurface | None]:
    # TODO: need pre filtering (ex. Exclude surface in completely different direction)
    collisions = []
    for suf in surfaces:
        c_param = calc_collision_param(suf, part)
        if do_collision(c_param, suf.get_basis_norm()):
            collisions.append((float(c_param[2]), suf))

    if len(collisions) == 0:
        return -1, None
    return min(collisions, key=lambda col: col[0])
