"""Dex Tactile v1: bounded speed costs, lower wrist cutoff and lateral goals."""

import math

from isaaclab.utils import configclass

from .assets.robot import ARM_JOINT_NAMES
from .mdp import goal_rewards
from .tactile_omni_renewal_env_cfg import InspireDexTactileSweepOmniRenewalEnvCfg


@configclass
class InspireDexTactileSweepOmniRenewalV1EnvCfg(InspireDexTactileSweepOmniRenewalEnvCfg):
    """Inherit v0's 57D/8D contract and specialize only the requested terms."""

    def __post_init__(self):
        super().__post_init__()
        # PhysX joint speed limits in rad/s, below the 5 rad/s failure cutoff.
        self.scene.robot.actuators["arm"].velocity_limit_sim = dict(
            zip(ARM_JOINT_NAMES, (2.0, 2.0, 2.0, 4.0, 4.0, 4.0))
        )
        self.rewards.pushing_speed.func = goal_rewards.exponential_pushing_speed_cost_limit
        self.rewards.pushing_speed.params["max_cost"] = 100.0
        self.rewards.arm_speed_excess.func = goal_rewards.exponential_joint_speed_cost_limit
        self.rewards.arm_speed_excess.params["max_cost"] = 100.0
        self.terminations.shelf_collision.params["wrist_clearance"] = 0.03
        self.events.object_spawn.params["excluded_x_half_angle"] = math.radians(15.0)
