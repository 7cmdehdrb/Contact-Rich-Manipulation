"""Whole-Hand contact and bounded physical changes for Fusion v1.0.

These are reward rates. RewardManager applies each weight and ``step_dt``
once. Object acceleration is read directly from Isaac Lab / PhysX.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase

from ..contact_sweep_clutter_fusion.safety_rewards import WrenchRatePenalty
from ..mdp.goal_rewards import _exponential_speed_excess_limit
from .contacts import hand_obstacle_contact_force_w, whole_hand_contact_force_w
from .object_state import active_object_mask


def hand_contact_any(env, threshold_n=0.01, goal_contact_fade_width=0.03):
    """Contact rate faded to zero at the current goal-hold boundary.

    As with the previous tactile ON reward, sustained contact earns the same
    rate on every step, without a duration ramp. Multiply that rate by
    ``clamp((XY_goal_distance - current_hold_tolerance) / fade_width, 0, 1)``.
    Read the live goal-hold term so curriculum updates affect both rewards
    together. The 17 tactile channels remain observation-only.
    """
    if not math.isfinite(threshold_n) or threshold_n <= 0.0:
        raise ValueError("threshold_n must be finite and positive")
    if not math.isfinite(goal_contact_fade_width) or goal_contact_fade_width <= 0.0:
        raise ValueError("goal_contact_fade_width must be finite and positive")
    force = whole_hand_contact_force_w(env)
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    valid = torch.isfinite(force).all(dim=-1) & env.sensor_data_fresh
    hold_params = env.reward_manager.get_term_cfg("goal_hold").params
    tolerance = hold_params["position_tolerance"]
    goal = env.command_manager.get_command(hold_params.get("command_name", "target_goal_pos"))
    rows = torch.arange(env.num_envs, device=env.device)
    position = env.scene["object_collection"].data.object_pos_w[rows, env.target_id[:, 0].long()]
    distance = torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1)
    alpha = ((distance - tolerance) / goal_contact_fade_width).clamp(0.0, 1.0)
    return torch.where(valid & (magnitude >= threshold_n), alpha, 0.0)


def obstacle_contact_force_cost(env, free_force=20.0, max_cost=100.0):
    """Exponential excess of net Hand force from non-target obstacle contacts.

    Contact identities are filtered by the sensors before summing their
    world-axis force vectors. Target contacts cannot contribute. The local
    joint-speed cost's ``0.5 * expm1(2 * excess)`` and finite cap are retained.
    """
    force = hand_obstacle_contact_force_w(env)
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    magnitude = torch.nan_to_num(magnitude, nan=math.inf, posinf=math.inf, neginf=math.inf)
    cost = _exponential_speed_excess_limit(magnitude, free_force, max_cost)
    return torch.where(env.sensor_data_fresh, cost, 0.0)


class WrenchMagnitudeChangePenalty(WrenchRatePenalty):
    """Bounded change of scalar force/moment magnitudes at the observed EEF.

    ``abs(norm(current) - norm(previous))`` has units N and Nm, respectively;
    rotating a vector at constant magnitude is free. The observation's exact
    compensated EEF sample supplies both the reference point and axes.

    Preserve Fusion's normalized exponential, maximum of the two severities,
    [0, 1] raw cap, and cumulative 5% earned-positive-reward budget. Saturation
    tracks each current free threshold at ratio 2.5, preserving the parent's
    saturation/free ratio throughout the new threshold curriculum.
    """

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_wrench[env_ids] = 0.0
        self._previous_valid[env_ids] = False
        self._cached_raw[env_ids] = 0.0
        # Preserve the physical-step key: a partial reset must not recompute
        # unaffected rows against their already-updated history in this step.

    def __call__(
        self, env, free_force_change=50.0, free_moment_change=5.0,
        saturation_ratio=2.5, sharpness=3.0, max_positive_fraction=0.05,
    ):
        for name, value in (("force", free_force_change), ("moment", free_moment_change)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"free_{name}_change must be finite and positive")
        if not math.isfinite(saturation_ratio) or saturation_ratio <= 1.0:
            raise ValueError("saturation_ratio must be finite and greater than one")
        if not math.isfinite(sharpness) or not 0.0 < sharpness <= 50.0:
            raise ValueError("sharpness must be finite and in (0, 50]")
        if not math.isfinite(max_positive_fraction) or not 0.0 <= max_positive_fraction <= 0.05:
            raise ValueError("max_positive_fraction must be in [0, 0.05]")

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
        return torch.minimum(self._cached_raw, self._reward_budget(env, max_positive_fraction))


class ObjectAccelerationPenalty(ManagerTermBase):
    """Sum independent exponential costs of every object's 3D acceleration.

    Read Isaac Lab's world-frame center-of-mass linear acceleration from
    PhysX (last physics step, not a control-interval velocity difference).
    Its Euclidean norm penalizes acceleration and deceleration symmetrically.
    Each active object has a 100-unit cap, so the sum is bounded by active
    object count times 100. Keep only reset/activation validity, not velocity
    history: the first sample after reset or activation is suppressed.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_name = cfg.params.get("asset_name", "object_collection")
        acceleration = env.scene[asset_name].data.object_com_lin_acc_w
        self._previous_valid = torch.zeros(acceleration.shape[:-1], dtype=torch.bool, device=acceleration.device)
        self._cached_cost = torch.zeros(env.num_envs, device=env.device)
        self._cached_key = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_valid[env_ids] = False
        self._cached_cost[env_ids] = 0.0

    def __call__(self, env, free_acceleration=0.3, max_cost=100.0, asset_name="object_collection"):
        if not math.isfinite(free_acceleration) or free_acceleration < 0.0:
            raise ValueError("free_acceleration must be finite and nonnegative")
        if not math.isfinite(max_cost) or max_cost <= 0.0:
            raise ValueError("max_cost must be finite and positive")
        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            acceleration_w = env.scene[asset_name].data.object_com_lin_acc_w
            active = active_object_mask(env, acceleration_w)
            acceleration = torch.linalg.vector_norm(acceleration_w, dim=-1)
            acceleration = torch.nan_to_num(
                acceleration, nan=math.inf, posinf=math.inf, neginf=math.inf,
            )
            costs = _exponential_speed_excess_limit(acceleration, free_acceleration, max_cost)
            self._cached_cost[:] = torch.where(active & self._previous_valid, costs, 0.0).sum(dim=-1)
            self._previous_valid[:] = active
            self._cached_key = key
        return self._cached_cost.clone()
