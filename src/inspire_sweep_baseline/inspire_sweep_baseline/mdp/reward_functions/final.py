"""Final v1 sums of existing costs, preserving component weights and caps."""

import math
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from .goal_rewards import action_rate_l2, joint_vel_l2
from .contact_sweep_clutter_fusion import combined_active_object_cost, ObjectAccelerationPenalty
from .sweep import ShelfCollisionPenalty
from ..termination_functions.sweep import shelf_collision


def action_rate_penalty(env):
    return (0.03 * action_rate_l2(env, action_group="arm", max_cost=100.0)
            + 0.03 * action_rate_l2(env, action_group="hand", max_cost=100.0))


def joint_velocity_penalty(env, arm_cfg: SceneEntityCfg, hand_cfg: SceneEntityCfg):
    return (0.03 * joint_vel_l2(env, asset_cfg=arm_cfg, max_cost=100.0)
            + 0.01 * joint_vel_l2(env, asset_cfg=hand_cfg, max_cost=100.0))


class ObjectPenalty(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.acceleration = ObjectAccelerationPenalty(cfg, env)
        self._acceleration_enabled = False

    def reset(self, env_ids=None):
        self.acceleration.reset(env_ids)

    def __call__(self, env, free_angle=math.radians(5), fail_angle=0.9,
                 displacement_ratio=5.0, displacement_max_cost=0.04,
                 free_acceleration=50.0, acceleration_weight=0.0, max_cost=100.0):
        cost = 30.0 * combined_active_object_cost(
            env, free_angle, fail_angle, displacement_ratio, displacement_max_cost)
        enabled = acceleration_weight != 0.0
        if enabled != self._acceleration_enabled:
            self.acceleration.reset()
            self._acceleration_enabled = enabled
        if enabled:
            cost = cost - acceleration_weight * self.acceleration(env, free_acceleration, max_cost)
        return cost


class ShelfPenalty(ShelfCollisionPenalty):
    def __call__(self, env):
        # The old event is collision-only, not the new collision OR movement
        # termination. Re-evaluate exactly that component with live thresholds.
        params = env.termination_manager.get_term_cfg('shelf_termination').params
        collision = shelf_collision(env, threshold=params.get('threshold', .1),
            check_fingers=params.get('check_fingers', False),
            wrist_clearance=params.get('wrist_clearance', .03))
        return 3.0 * collision.float() / env.step_dt + 0.5 * super().__call__(env)
