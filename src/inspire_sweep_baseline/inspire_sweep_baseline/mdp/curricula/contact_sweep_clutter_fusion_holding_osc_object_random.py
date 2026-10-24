"""Holding OSC with a cylinder target and fourteen upright NVIDIA shelf props.

Holding OSC v2 with randomized props growing from 1/1 to 8/8."""

from .contact_sweep_clutter_fusion_holding import FusionHoldingV0CurriculumCfg
from .contact_sweep_clutter_fusion_holding import FusionHoldingV2CurriculumCfg
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg
from ..curriculum_functions.contact_sweep_clutter_fusion_holding_osc_object_random import gray_zone_padding_schedule

@configclass
class ObjectRandomCurriculumCfg(FusionHoldingV0CurriculumCfg):
    obstacle_population = None  # Temporary max-population test mode.


@configclass
class ObjectRandomV1CurriculumCfg(FusionHoldingV2CurriculumCfg):
    obstacle_population = FusionHoldingV2CurriculumCfg().obstacle_population.replace(
        params={**FusionHoldingV2CurriculumCfg().obstacle_population.params,
                "final_gray": 8, "final_white": 8},
    )


@configclass
class ObjectRandomV2CurriculumCfg(ObjectRandomV1CurriculumCfg):
    obstacle_population = ObjectRandomV1CurriculumCfg().obstacle_population.replace(
        params={**ObjectRandomV1CurriculumCfg().obstacle_population.params,
                "final_gray": 6, "final_white": 6,
                "white_start_iteration": 3000, "gray_start_iteration": 4000, "interval": 2000},
    )
    gray_zone_padding = CurriculumTermCfg(
        func=gray_zone_padding_schedule,
        params={"points": ((6000, 0.0), (12000, 0.15)), "interval": 250},
    )
