"""Shrink the common goal radius linearly at each learning iteration."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v0.curriculum import reward_schedule
from ..contact_sweep_clutter_fusion_holding_v0.curriculum_cfg import FusionHoldingV0CurriculumCfg


@configclass
class FusionHoldingV2CurriculumCfg(FusionHoldingV0CurriculumCfg):
    goal_position_scale = None
    goal_position_tolerance = CurriculumTermCfg(
        func=reward_schedule,
        params={"term_name": "goal_hold", "parameter": "position_tolerance",
                "points": ((0, 0.02), (5000, 0.02), (10000, 0.01), (20000, 0.005)),
                "interval": 1},
    )
