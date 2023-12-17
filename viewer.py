import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np

import base_geom
import camera


def _plot_vector_3d(axes: plt.Axes, loc, vec, color: str):
    axes.quiver(loc[0], loc[1], loc[2], vec[0], vec[1], vec[2],
                color=color, length=1, arrow_length_ratio=0.2)


def _set_lims(ax: plt.Axes, vectors: np.ndarray, locations: np.ndarray):
    ends = np.r_[locations, vectors + locations]
    mins = np.min(ends, axis=0)
    maxs = np.max(ends, axis=0)

    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_zlabel("z", fontsize=16)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.grid()


def vector(vectors: np.ndarray, locations: np.ndarray, colors: list[str], surfaces: list[np.ndarray]):
    assert vectors.ndim == 2
    assert locations.ndim == 2

    # FigureとAxes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    _set_lims(ax, vectors, locations)

    for v, loc, col in zip(vectors, locations, colors):
        _plot_vector_3d(ax, loc, v, col)

    for suf in surfaces:
        ax.add_collection3d(art3d.Poly3DCollection([suf], alpha=0.3))

    plt.show()


def plot_color_arr(colors: list[np.ndarray]):
    size = 20
    arr = np.ones((size, size * len(colors), 3), dtype=np.uint8)

    for ci in range(len(colors)):
        color = (colors[ci] * 255).astype(np.uint8)
        arr[:, ci * size: (ci + 1) * size, :] *= color[np.newaxis, np.newaxis, :]

    plt.imshow(arr)
    plt.show()


def show(particles: list[base_geom.BaseParticle], cam: camera.Camera, gamma: float = 1.0):
    color_arr_f4 = np.array([p.get_light().to_color().get_array() for p in particles])
    # gamma correction
    color_arr = (np.power(color_arr_f4, gamma) * 255).astype(np.uint8)
    color_arr = color_arr.reshape((cam.pixel_v, cam.pixel_h, 3))
    plt.imshow(color_arr[::-1])
    plt.show()
