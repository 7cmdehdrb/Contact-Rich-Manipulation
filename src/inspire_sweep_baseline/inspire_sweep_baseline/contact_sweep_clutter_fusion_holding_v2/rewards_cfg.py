"""Double settling weight and grant contact credit inside its shared radius."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_holding_v0.rewards_cfg import FusionHoldingV0RewardsCfg
from .mdp import goal_settling, hand_contact_or_goal


@configclass
class FusionHoldingV2RewardsCfg(FusionHoldingV0RewardsCfg):
    goal_hold = FusionHoldingV0RewardsCfg().goal_hold.replace(
        func=goal_settling, weight=4.0,
        params={"position_tolerance": 0.02, "speed_scale": 0.05},
    )
    hand_contact = FusionHoldingV0RewardsCfg().hand_contact.replace(
        func=hand_contact_or_goal, params={"threshold_n": 0.01},
    )
