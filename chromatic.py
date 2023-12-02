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
        return CLight(add_chromatic(self.get_array(), other_color.to_light().get_array()))


def add_chromatic(chr_elem1: np.ndarray, chr_elem2: np.ndarray) -> np.ndarray:
    return chr_elem1 * chr_elem2


def add_chromatic_multi(chr_elements: list[np.ndarray]) -> np.ndarray:
    return np.prod(chr_elements, axis=0)
