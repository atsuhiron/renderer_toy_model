import numpy as np


class Color:
    def __init__(self, color: np.ndarray):
        self._color = color.astype(np.float32)

    def get_array(self) -> np.ndarray:
        return self._color

    def as_uint8(self) -> np.ndarray:
        return (self._color * 255).astype(np.uint8)


def add_color(col1: Color, col2: Color) -> np.ndarray:
    return col1.get_array() * col2.get_array()


def add_light(col1: Color, col2: Color) -> np.ndarray:
    return 1 - (1 - col1.get_array()) * (1 - col2.get_array())
