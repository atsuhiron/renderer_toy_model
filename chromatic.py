from __future__ import annotations

import numpy as np
import numba

from const import NUMBA_OPT


class Chromatic:
    def __init__(self, elements: np.ndarray | list[float | int] | str):
        self.elements = self._from_array_or_code(elements)

    def get_array(self) -> np.ndarray:
        return self.elements

    def as_uint8(self) -> np.ndarray:
        return (self.elements * 255).astype(np.uint8)

    @staticmethod
    def _from_array_or_code(value: np.ndarray | list[float | int] | str) -> np.ndarray:
        if isinstance(value, np.ndarray):
            # np.array([23, 128, 128])
            arr = value.astype(np.float32)
        elif isinstance(value, list):
            # [0.23, 0.85, 0.0]
            arr = np.array(value, dtype=np.float32)
        elif isinstance(value, str):
            # #FF80A5
            assert value[0] == "#"
            assert len(value) == 7
            arr = np.array([int(value[1:3], base=16),
                            int(value[3:5], base=16),
                            int(value[5:7], base=16)], dtype=np.float32)
        else:
            raise TypeError("Not supported type")

        if np.max(arr) > 1:
            arr /= 255
        return arr


class CColor(Chromatic):
    def to_light(self) -> CLight:
        return CLight(1 - self.get_array())


class CLight(Chromatic):
    def to_color(self) -> CColor:
        return CColor(1 - self.get_array())

    def add_color(self, other_color: CColor) -> CLight:
        return CColor(add_color([self.to_color().get_array(), other_color.get_array()])).to_light()


def add_color(chr_elements: list[np.ndarray]) -> np.ndarray:
    return np.prod(chr_elements, axis=0)


@numba.jit(**NUMBA_OPT)
def add_light(light_elements: np.ndarray, intensities: np.ndarray) -> tuple[np.ndarray, float]:
    color = (1 - light_elements) * intensities[:, np.newaxis]
    dark_mask = (np.sum(color, axis=1) > 0).astype(np.float32)
    masked_intensity = np.sum(dark_mask * intensities)
    return 1 - np.sum(color, axis=0), masked_intensity


if __name__ == "__main__":
    from viewer import plot_color_arr

    base_col = CColor(np.array([0.1, 0.1, 0.5]))
    l1 = CLight(np.array([0.00, 0.05, 0.05]))
    l1_on_base = l1.add_color(base_col)
    plot_color_arr([base_col.get_array(), l1.to_color().get_array(), l1_on_base.to_color().get_array()])

    l2 = CLight(np.array([1, 1, 0]))
    l3 = CLight(np.array([0, 1, 1]))
    l23_ele, itst = add_light(np.array([l2.get_array(), l3.get_array()]), np.array([1, 1]))
    l23 = CLight(l23_ele)
    plot_color_arr([light.to_color().get_array() for light in [l2, l3, l23]])
