"""Analytic fixture generation/evaluation only. Never imported by estimator."""
from dataclasses import dataclass
import math
import numpy as np
from observed_obb import Box, Observation, observe
from observed_obb.observation import _footprint
from .shapes import SHAPES


@dataclass(frozen=True)
class Case:
    name: str
    shape: str
    pattern: str
    truth: Box
    scene: tuple[Box, ...]
    observations: tuple[Observation, ...]
    target_id: int


def make_case(shape_index, pattern, *, scale=1., yaw=0., translation=(0., 0.)):
    shape = SHAPES[shape_index]
    size = tuple(scale * v for v in shape.size)
    truth = Box((*translation, size[2]/2), size, yaw)
    footprint = _footprint(truth)
    width = float(np.ptp(footprint[:, 1]))
    intervals = {"clear": (), "left": ((-.9, .05),), "right": ((-.05, .9),),
                 "both": ((-.9, -.2), (.2, .9)), "split": ((-.12, .12),),
                 "rear": ((-.05, .9),)}[pattern]
    others = tuple(Box((translation[0] + (-.2 if pattern == "rear" else .2),
                        translation[1] + (lo+hi)*width/2, .03),
                       (.035, (hi-lo)*width, .06)) for lo, hi in intervals)
    scene = (truth, *others)
    observations = observe(scene)
    # Fixture construction knows target is source 0. Only snapshot and local ID
    # cross into inference; shape_index, scene and truth remain evaluation data.
    if not observations or observations[0].box.size[2] != truth.size[2]:
        raise ValueError("Fixture target was filtered out")
    return Case(f"{shape.name}_{pattern}_yaw{math.degrees(yaw):g}", shape.name,
                pattern, truth, scene, observations, observations[0].observation_id)


def standard_cases():
    scales = (.94, 1.04, .98, 1.06, 1.02, .96, 1.08, 1.)
    cases = [make_case(i, pattern, scale=scales[i]) for i in range(8)
             for pattern in ("clear", "left", "right", "both")]
    cases.extend((make_case(1, "right", yaw=math.radians(30)),
                  make_case(2, "both", yaw=math.radians(30)),
                  make_case(0, "rear"), make_case(0, "split")))
    return cases
