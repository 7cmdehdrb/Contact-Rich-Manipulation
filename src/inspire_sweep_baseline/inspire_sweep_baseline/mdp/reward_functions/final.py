"""Final v1 cost mergers and v1-1 goal proximity reward."""

import math
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from .goal_rewards import _target_state, action_rate_l2, joint_vel_l2
from .contact_sweep_clutter_fusion import combined_active_object_cost, ObjectAccelerationPenalty
from .sweep import ShelfCollisionPenalty
from ..termination_functions.sweep import shelf_collision
from ..termination_functions.goal_terminations import _object_position
from ..common.contact_sweep_clutter_fusion_holding import target_measurements
from ..common.contact_sweep_clutter_fusion_holding import goal_radius
from ..common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w


def exponential_planar_goal_proximity(env, command_name="target_goal_pos", distance_scale=0.05):
    """Dense XY goal reward; distance_scale is the e-folding distance in meters."""
    if not math.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("distance_scale must be finite and positive")
    goal = env.command_manager.get_command(command_name)[:, :2]
    target = _target_state(env)[:, :2]
    distance = torch.linalg.vector_norm(target - goal, dim=-1)
    return torch.exp(-distance / distance_scale)


def goal_accuracy_settling_centered(env, position_tolerance=0.01, speed_scale=0.05,
                                    command_name="target_goal_pos"):
    """Centered hold reward only for a valid target inside the inclusive goal."""
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance, speed_scale)):
        raise ValueError("Goal radius and speed scale must be finite and positive")
    distance, speed, valid = target_measurements(env, command_name)
    score = torch.exp(-math.log(2) * ((distance / position_tolerance) ** 2
                                      + (speed / speed_scale) ** 2))
    center_bonus = (1.0 - distance / position_tolerance).clamp(0.0, 1.0).square()
    return torch.where(valid & (distance <= position_tolerance),
                       score * (1.0 + center_bonus), 0.0)


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
    def __call__(self, env, termination_term_name='shelf_termination'):
        # The old event is collision-only, not the new collision OR movement
        # termination. Re-evaluate exactly that component with live thresholds.
        params = env.termination_manager.get_term_cfg(termination_term_name).params
        collision = shelf_collision(env, threshold=params.get('threshold', .1),
            check_fingers=params.get('check_fingers', False),
            wrist_clearance=params.get('wrist_clearance', .03))
        return 3.0 * collision.float() / env.step_dt + 0.5 * super().__call__(env)


def goal_accuracy_outer_settling(env, position_tolerance=0.02, outer_radius=0.06,
                                 speed_scale=0.05, command_name="target_goal_pos"):
    """v0_1 raw hold rate: centered inner score joined to a quadratic outer tail."""
    if not all(math.isfinite(v) and v > 0 for v in
               (position_tolerance, outer_radius, speed_scale)) or outer_radius <= position_tolerance:
        raise ValueError("Expected 0 < goal radius < outer radius and positive speed scale")
    distance, speed, valid = target_measurements(env, command_name)
    inner = torch.exp(-math.log(2) * (distance / position_tolerance).square())
    inner *= 1.0 + (1.0 - distance / position_tolerance).clamp(0.0, 1.0).square()
    outer = 0.5 * ((outer_radius - distance) / (outer_radius - position_tolerance)).clamp(0.0, 1.0).square()
    rate = torch.where(distance <= position_tolerance, inner, outer)
    rate *= torch.exp(-math.log(2) * (speed / speed_scale).square())
    return torch.where(valid, rate, 0.0)


def target_contact_or_goal(env, threshold_n=0.01, goal_hold_term_name="target_goal_hold"):
    """Credit target/hand resultant force or valid inclusive goal occupancy once."""
    if not math.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("Contact threshold must be finite and positive")
    force = target_hand_contact_force_w(env)
    contact = (env.sensor_data_fresh & torch.isfinite(force).all(-1)
               & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n))
    hold = env.reward_manager.get_term_cfg(goal_hold_term_name)
    distance, _, valid = target_measurements(env, hold.params.get('command_name', 'target_goal_pos'))
    in_goal = valid & (distance <= goal_radius(env, goal_hold_term_name))
    return (contact | in_goal).to(force.dtype)


def target_contact_only(env, threshold_n=0.01):
    """Reward fresh target/hand normal resultant contact, without a goal fallback."""
    if not math.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("Contact threshold must be finite and positive")
    force = target_hand_contact_force_w(env)
    return (env.sensor_data_fresh & torch.isfinite(force).all(-1)
            & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n)).to(force.dtype)


def target_goal_timeout_bonus(env, position_tolerance=0.02, command_name="target_goal_pos"):
    """Unit terminal event for strict XY proximity, independent of the step size.

    The timeout transition is rewarded before automatic reset, so each episode
    can receive this event only once. No speed, height, or hold-time gate applies.
    """
    goal = env.command_manager.get_command(command_name)[:, :2]
    target = _object_position(env)[:, :2]
    distance = torch.linalg.vector_norm(target - goal, dim=-1)
    # TerminationManager has already evaluated this step, before rewards/reset.
    success = env.termination_manager.get_term("target_goal_success_at_timeout")
    reached_at_timeout = success & (distance < position_tolerance)
    # RewardManager multiplies by weight * step_dt: weight=30 pays exactly +30.
    return reached_at_timeout.float() / env.step_dt
