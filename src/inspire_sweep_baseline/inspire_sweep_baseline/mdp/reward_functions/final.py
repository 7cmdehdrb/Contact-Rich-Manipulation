"""Final v1 cost mergers and v1-1 goal proximity reward."""

import math
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from .goal_rewards import _target_state, action_rate_l2, joint_vel_l2
from .contact_sweep_clutter_fusion import combined_active_object_cost, ObjectAccelerationPenalty
from .sweep import ShelfCollisionPenalty
from ..termination_functions.sweep import shelf_collision
from ..common.contact_sweep_clutter_fusion_holding import target_measurements, goal_radius
from ..common.contact_sweep_clutter_fusion_contacts import whole_hand_contact_force_w


def exponential_planar_goal_proximity(env, command_name="target_goal_pos", distance_scale=0.05):
    """Dense XY goal reward; distance_scale is the e-folding distance in meters."""
    if not math.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("distance_scale must be finite and positive")
    goal = env.command_manager.get_command(command_name)[:, :2]
    target = _target_state(env)[:, :2]
    distance = torch.linalg.vector_norm(target - goal, dim=-1)
    return torch.exp(-distance / distance_scale)


def strict_goal_settling_centered(env, position_tolerance=0.01, speed_scale=0.05,
                                  command_name="target_goal_pos"):
    """Centered hold reward only for a valid target strictly inside the goal."""
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance, speed_scale)):
        raise ValueError("Goal radius and speed scale must be finite and positive")
    distance, speed, valid = target_measurements(env, command_name)
    score = torch.exp(-math.log(2) * ((distance / position_tolerance) ** 2
                                      + (speed / speed_scale) ** 2))
    center_bonus = (1.0 - distance / position_tolerance).clamp(0.0, 1.0).square()
    return torch.where(valid & (distance < position_tolerance),
                       score * (1.0 + center_bonus), 0.0)


def strict_hand_contact_or_goal(env, threshold_n=0.01):
    """Keep measured contact credit; require strict goal occupancy for its proxy."""
    if not math.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("Contact threshold must be finite and positive")
    force = whole_hand_contact_force_w(env)
    contact = (torch.isfinite(force).all(-1) & env.sensor_data_fresh
               & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n))
    goal = env.reward_manager.get_term_cfg("goal_hold")
    distance, _, valid = target_measurements(env, goal.params.get("command_name", "target_goal_pos"))
    return (contact | (valid & (distance < goal_radius(env)))).to(force.dtype)


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
