from __future__ import annotations

import numpy as np


class Chromatic:
    def __init__(self, elements: np.ndarray):
        self._elements = elements

    def get_array(self) -> np.ndarray:
        return self._elements

    def as_uint8(self) -> np.ndarray:
        return (self._elements * 255).astype(np.uint8)


class CColor(Chromatic):
    def to_light(self) -> CLight:
        return CLight(1 - self.get_array())


class CLight(Chromatic):
    def to_color(self) -> CColor:
        return CColor(1 - self.get_array())

    def add_color(self, other_color: CColor) -> CLight:
        return CColor(add_chromatic(self.to_color().get_array(), other_color.get_array())).to_light()


def add_chromatic(chr_elem1: np.ndarray, chr_elem2: np.ndarray) -> np.ndarray:
    return chr_elem1 * chr_elem2


def add_chromatic_multi(chr_elements: list[np.ndarray]) -> np.ndarray:
    return np.prod(chr_elements, axis=0)


if __name__ == "__main__":
    from viewer import plot_color_arr

    base_col = CColor(np.array([0.1, 0.1, 0.2]))
    l1 = CLight(np.array([0.00, 0.05, 0.05]))
    l1_on_base = l1.add_color(base_col)
    plot_color_arr([base_col.get_array(), l1.to_color().get_array(), l1_on_base.to_color().get_array()])

    l2 = CLight(np.array([1, 1, 0]))
    l3 = CLight(np.array([0, 1, 1]))
    l23 = CLight(add_chromatic_multi([l2.get_array(), l3.get_array()]))
    plot_color_arr([light.to_color().get_array() for light in [l2, l3, l23]])
