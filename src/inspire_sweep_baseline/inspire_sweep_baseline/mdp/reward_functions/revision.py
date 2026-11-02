"""Revision hold, contact-height gate and inclusive timeout bonus."""

import math

import torch
from isaaclab.managers import ManagerTermBase

from ..termination_functions.goal_terminations import _object_position

from ..common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
from ..common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w
from ..common.revision_goal import GOAL_SUCCESS_RADIUS, HOLD_OUTER_RADIUS, target_measurements


def goal_hold(env, position_tolerance=GOAL_SUCCESS_RADIUS, outer_radius=HOLD_OUTER_RADIUS,
              speed_scale=0.05, command_name="target_goal_pos"):
    """The v0_2 continuous settling curve with bounds-aware bottom validation."""
    if not all(math.isfinite(v) and v > 0 for v in
               (position_tolerance, outer_radius, speed_scale)) or outer_radius <= position_tolerance:
        raise ValueError("Expected 0 < goal radius < outer radius and positive speed scale")
    if position_tolerance != env.cfg.goal_success_radius:
        raise ValueError("Revision Hold and success radius must remain synchronized")
    distance, speed, valid = target_measurements(env, command_name)
    inner = torch.exp(-math.log(2) * (distance / position_tolerance).square())
    inner *= 1.0 + (1.0 - distance / position_tolerance).clamp(0.0, 1.0).square()
    outer = 0.5 * ((outer_radius - distance) / (outer_radius - position_tolerance)).clamp(0.0, 1.0).square()
    rate = torch.where(distance <= position_tolerance, inner, outer)
    rate *= torch.exp(-math.log(2) * (speed / speed_scale).square())
    return torch.where(valid, rate, 0.0)


def target_contact_below_height(env, threshold_n=0.01, frame_name="ee_frame"):
    """Require actual fresh target/hand contact AND EEF below one fixed height.

    The work-surface reference is captured before physics starts. Neither the
    target shape/scale/pose nor later shelf motion changes the height threshold.
    Equality and non-finite EEF heights fail the strict height predicate.
    """
    height_limit = float(env.cfg.contact_height_limit)
    if not math.isfinite(threshold_n) or threshold_n <= 0:
        raise ValueError("Contact threshold must be finite and positive")
    if not math.isfinite(height_limit) or height_limit <= 0:
        raise ValueError("Contact height limit must be finite and positive")
    force = target_hand_contact_force_w(env)
    eef_z = env.scene[frame_name].data.target_pos_w[:, 0, 2]
    ceiling = env._revision_initial_shelf_pos[:, 2] + SHELF_SURFACE_HEIGHT + height_limit
    contact = (
        env.sensor_data_fresh
        & torch.isfinite(force).all(-1)
        & (torch.linalg.vector_norm(force, dim=-1) >= threshold_n)
        & torch.isfinite(eef_z)
        & (eef_z < ceiling)
    )
    return contact.to(force.dtype)


def successful_timeout_bonus(env):
    """Reward the same inclusive success partition exactly once, before reset."""
    return env.termination_manager.get_term("target_goal_success_at_timeout").float() / env.step_dt


def goal_hold_v1(env, position_tolerance=GOAL_SUCCESS_RADIUS, free_speed=0.05,
                 speed_scale=0.05, command_name="target_goal_pos"):
    """Bounded centered Gaussian inside the goal, with a free-speed plateau."""
    if not all(math.isfinite(v) and v > 0 for v in (position_tolerance, speed_scale)):
        raise ValueError("Goal radius and speed scale must be finite and positive")
    if not math.isfinite(free_speed) or free_speed < 0:
        raise ValueError("Free speed must be finite and nonnegative")
    if position_tolerance != env.cfg.goal_success_radius:
        raise ValueError("Revision Hold and success radius must remain synchronized")
    distance, speed, valid = target_measurements(env, command_name)
    position = 2.0 * torch.exp(-4.0 * math.log(2) * (distance / position_tolerance).square())
    damping = torch.exp(-math.log(2) * ((speed - free_speed).clamp_min(0) / speed_scale).square())
    return torch.where(valid & (distance < position_tolerance), position * damping, 0.0)


class FirstGoalEntryBonus(ManagerTermBase):
    """Pay once per episode at a sampled strict XY entry, including failure steps."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._entered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._cached_event = torch.zeros(env.num_envs, device=env.device)
        self._cached_key = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._entered[env_ids] = False
        self._cached_event[env_ids] = 0.0
        # Keep unaffected rows' cached result and history on partial reset.

    def __call__(self, env, command_name="target_goal_pos"):
        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            goal = env.command_manager.get_command(command_name)
            distance = torch.linalg.vector_norm(_object_position(env)[:, :2] - goal[:, :2], dim=-1)
            inside = torch.isfinite(distance) & (distance < env.cfg.goal_success_radius)
            self._cached_event[:] = (inside & ~self._entered).float()
            self._entered |= inside
            self._cached_key = key
        return self._cached_event / env.step_dt


class IndependentWrenchMagnitudeChangePenalty(ManagerTermBase):
    """Revision-v1 wrench cost: the v0 raw curve without any reward budget.

    The v0 term remains untouched. Sensor sampling, invalid history handling,
    first-sample suppression and partial-reset caching match that term.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_wrench = torch.zeros((env.num_envs, 6), device=env.device)
        self._previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._cached_raw = torch.zeros(env.num_envs, device=env.device)
        self._cached_key = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_wrench[env_ids] = 0.0
        self._previous_valid[env_ids] = False
        self._cached_raw[env_ids] = 0.0

    def __call__(
        self, env, free_force_change=20.0, free_moment_change=2.0,
        saturation_ratio=2.5, sharpness=3.0,
    ):
        for name, value in (("force", free_force_change), ("moment", free_moment_change)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"free_{name}_change must be finite and positive")
        if not math.isfinite(saturation_ratio) or saturation_ratio <= 1.0:
            raise ValueError("saturation_ratio must be finite and greater than one")
        if not math.isfinite(sharpness) or not 0.0 < sharpness <= 50.0:
            raise ValueError("sharpness must be finite and in (0, 50]")

        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            term = env.observation_manager.cfg.policy.wrist_wrench
            current = term.func(env, **term.params)
            fresh = env.sensor_data_fresh
            finite = torch.isfinite(current).all(dim=-1)
            force_change = (
                torch.linalg.vector_norm(current[:, :3], dim=-1)
                - torch.linalg.vector_norm(self._previous_wrench[:, :3], dim=-1)
            ).abs()
            moment_change = (
                torch.linalg.vector_norm(current[:, 3:], dim=-1)
                - torch.linalg.vector_norm(self._previous_wrench[:, 3:], dim=-1)
            ).abs()
            force_excess = (force_change - free_force_change).clamp_min(0.0) / (
                (saturation_ratio - 1.0) * free_force_change
            )
            moment_excess = (moment_change - free_moment_change).clamp_min(0.0) / (
                (saturation_ratio - 1.0) * free_moment_change
            )
            severity = torch.maximum(force_excess, moment_excess)
            severity = torch.nan_to_num(severity, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            raw = (torch.expm1(sharpness * severity) / math.expm1(sharpness)).clamp(0.0, 1.0)
            self._cached_raw[:] = torch.where(fresh & self._previous_valid, raw, 0.0)
            self._previous_wrench[:] = torch.where((fresh & finite)[:, None], current, 0.0)
            self._previous_valid[:] = fresh & finite
            self._cached_key = key
        return self._cached_raw.clone()
