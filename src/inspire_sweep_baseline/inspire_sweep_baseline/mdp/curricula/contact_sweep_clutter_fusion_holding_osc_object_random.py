"""Holding OSC with a cylinder target and fourteen upright NVIDIA shelf props.

Holding OSC v2 with randomized props growing from 1/1 to 8/8."""

from .contact_sweep_clutter_fusion_holding import FusionHoldingV0CurriculumCfg
from .contact_sweep_clutter_fusion_holding import FusionHoldingV2CurriculumCfg
from isaaclab.utils import configclass

@configclass
class ObjectRandomCurriculumCfg(FusionHoldingV0CurriculumCfg):
    obstacle_population = None  # Temporary max-population test mode.


@configclass
class ObjectRandomV1CurriculumCfg(FusionHoldingV2CurriculumCfg):
    obstacle_population = FusionHoldingV2CurriculumCfg().obstacle_population.replace(
        params={**FusionHoldingV2CurriculumCfg().obstacle_population.params,
                "final_gray": 8, "final_white": 8},
    )
