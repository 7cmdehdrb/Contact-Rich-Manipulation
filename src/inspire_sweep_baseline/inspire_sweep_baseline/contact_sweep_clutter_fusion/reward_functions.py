"""Exact reward aggregation using the original local cost functions."""

import math

from ..mdp.goal_rewards import action_rate_l2, joint_vel_l2
from ..contact_sweep_clutter.mdp import all_object_tilt_cost
from ..contact_sweep_clutter_v1.mdp import obstacle_displacement_l2


def combined_action_rate(env, max_cost=100.0):
    """Preserve each group's output cap before adding raw-action square sums."""
    return (action_rate_l2(env, action_group="arm", max_cost=max_cost)
            + action_rate_l2(env, action_group="hand", max_cost=max_cost))


def combined_joint_velocity(env, arm_cfg, hand_cfg, hand_ratio=1.0 / 3.0, max_cost=100.0):
    """With weight -0.03, retain Arm -0.03 and Hand -0.01 contributions."""
    return (joint_vel_l2(env, asset_cfg=arm_cfg, max_cost=max_cost)
            + hand_ratio * joint_vel_l2(env, asset_cfg=hand_cfg, max_cost=max_cost))


def combined_object_cost(
    env, free_angle=math.radians(5.0), fail_angle=0.9,
    displacement_ratio=2.5, displacement_max_cost=0.04,
):
    """Weight -2 reproduces -2 tilt and -5 displacement; -30 scales both x15."""
    return (
        all_object_tilt_cost(env, free_angle=free_angle, fail_angle=fail_angle)
        + displacement_ratio * obstacle_displacement_l2(env, max_cost=displacement_max_cost)
    )
