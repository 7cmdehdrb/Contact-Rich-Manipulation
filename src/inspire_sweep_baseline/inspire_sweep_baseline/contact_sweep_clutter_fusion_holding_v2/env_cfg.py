"""Holding v1 with synchronized goal tolerance and contact credit."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v1.env_cfg import FusionHoldingV1EnvCfg
from .curriculum_cfg import FusionHoldingV2CurriculumCfg
from .rewards_cfg import FusionHoldingV2RewardsCfg
from .terminations_cfg import FusionHoldingV2TerminationsCfg


@configclass
class FusionHoldingV2EnvCfg(FusionHoldingV1EnvCfg):
    rewards: FusionHoldingV2RewardsCfg = FusionHoldingV2RewardsCfg()
    terminations: FusionHoldingV2TerminationsCfg = FusionHoldingV2TerminationsCfg()
    curriculum: FusionHoldingV2CurriculumCfg = FusionHoldingV2CurriculumCfg()
