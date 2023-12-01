import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np


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
