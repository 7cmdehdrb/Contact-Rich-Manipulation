"""Predeclared population prior, never selected from evaluation ground truth."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Shape:
    name: str
    size: tuple[float, float, float]

    def __post_init__(self):
        if len(self.size) != 3 or not all(math.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("Shape sizes must be three positive finite lengths")


def _normalized(size):
    # Same envelope rule as the existing environment; input dimensions below
    # are declared synthetic proxies, NOT measured NVIDIA USD bounds.
    scale = 1.35 * min(.08 / math.hypot(*size[:2]), .12 / size[2])
    return tuple(scale * v for v in size)


SHAPES = (
    Shape("cylinder", (.08, .08, .10)),
    Shape("cube", (.06, .08, .10)),
    Shape("mug", (.101, .07, .10)),
    Shape("bottle", (.06, .06, .135)),
    Shape("003_cracker_box", _normalized((.06, .16, .21))),
    Shape("004_sugar_box", _normalized((.04, .09, .17))),
    Shape("005_tomato_soup_can", _normalized((.066, .066, .101))),
    Shape("002_master_chef_can", _normalized((.102, .102, .14))),
)
