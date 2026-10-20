"""Explicit starting values on top of the current Clutter v1 rewards."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.rewards_cfg import ContactSweepClutterV1RewardsCfg


@configclass
class ContactSweepClutterV1CurriculumRewardsCfg(ContactSweepClutterV1RewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.pushing_speed.params["free_speed"] = 0.1
        self.arm_speed_excess.params["free_speed"] = 1.0
        self.arm_action_rate.weight = -0.03
        self.arm_joint_velocity.weight = -0.03
