"""Only the existing tilt penalty expands from the target to all objects.

Initial coefficients calibrated against the 2026-10-20_14-04-31 v0 run.

Explicit starting values on top of the current Clutter v1 rewards.

Clutter v1 rewards plus a small, strictly budgeted wrench-change penalty."""

from ..reward_functions.contact_sweep_clutter import WrenchChangePenalty
from ..reward_functions.contact_sweep_clutter import all_object_tilt_cost
from ..reward_functions.contact_sweep_clutter import contact_any
from ..reward_functions.contact_sweep_clutter import contact_count
from ..reward_functions.contact_sweep_clutter import goal_hold_xy
from ..reward_functions.contact_sweep_clutter import obstacle_displacement_l2
from ..reward_functions.contact_sweep_clutter import shelf_failure
from .contact_sweep import ContactSweepRewardsCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepClutterRewardsCfg(ContactSweepRewardsCfg):
    def __post_init__(self):
        self.object_tilt.func = all_object_tilt_cost


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


@configclass
class ContactSweepClutterV1CurriculumRewardsCfg(ContactSweepClutterV1RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.pushing_speed.params["free_speed"] = 0.1
        self.arm_speed_excess.params["free_speed"] = 1.0
        self.arm_action_rate.weight = -0.03
        self.arm_joint_velocity.weight = -0.03


@configclass
class ContactSweepClutterV2RewardsCfg(ContactSweepClutterV1RewardsCfg):
    # Must remain after the inherited positive terms: their current-step sums
    # supply the budget. At 50 Hz, ignore changes <=0.5 N and <=0.01 N m.
    # Max -0.1/s, hence -1 over 10 s, AND <=5% of actual gross positive return.
    wrench_change = RewardTermCfg(
        func=WrenchChangePenalty,
        weight=-0.1,
        params={
            "free_force_change": 0.5,
            "free_moment_change": 0.01,
            "force_change_scale": 5.0,
            "moment_change_scale": 0.1,
            "max_positive_fraction": 0.05,
        },
    )
