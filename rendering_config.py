import dataclasses


@dataclasses.dataclass
class RenderingConfig:
    max_generation: int
    rough_surface_child_num: int
