"""Initial coefficients calibrated against the 2026-10-20_14-04-31 v0 run."""

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from ..contact_sweep_clutter.rewards_cfg import ContactSweepClutterRewardsCfg
from .mdp import (
    contact_any,
    contact_count,
    goal_hold_xy,
    obstacle_displacement_l2,
    shelf_failure,
)


@configclass
class ContactSweepClutterV1RewardsCfg(ContactSweepClutterRewardsCfg):
    # Five seconds in goal -> +4.5; seven seconds at >=5 contacts -> +2.24.
    # Together with logged progress: approximately 35/35/20/10 positive shares.
    goal_hold = RewardTermCfg(
        func=goal_hold_xy,
        weight=0.9,
        params={"position_tolerance": 0.01},
    )
    contact_any = RewardTermCfg(func=contact_any, weight=0.16)
    contact_count = RewardTermCfg(
        func=contact_count,
        weight=0.16,
        params={"max_contacts": 5},
    )
    # Two obstacles displaced 10 cm for 10 s -> -1.0; absolute cap -> -2.0/10 s.
    obstacle_displacement = RewardTermCfg(
        func=obstacle_displacement_l2,
        weight=-5.0,
        params={"max_cost": 0.04},
    )
    # Terminal-transition event: mdp cancels dt, so the actual cost is -3.
    # It is included in the final reward before the environment auto-resets.
    shelf_failure = RewardTermCfg(func=shelf_failure, weight=-3.0)

    def __post_init__(self):
        super().__post_init__()
        self.reaching.weight = 30.0
        self.goal_distance.weight = 100.0
        self.pushing_speed.params["free_speed"] = 0.1
        # self.arm_speed_excess.params["free_speed"] = 0.7
