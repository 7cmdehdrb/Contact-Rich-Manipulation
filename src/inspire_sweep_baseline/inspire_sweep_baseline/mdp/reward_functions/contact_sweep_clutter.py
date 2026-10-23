"""Clutter-specific observations, rewards, and failure conditions.

World-XY goal occupancy, shared tactile rewards, and bounded scene movement.

Reset-safe wrench-change cost with an episode-relative reward budget."""

from __future__ import annotations

from ..common.contact_sweep_clutter import all_object_tilt_angles
from isaaclab.managers import ManagerTermBase
import math as math
import torch as torch

def all_object_tilt_cost(env, free_angle=math.radians(5.0), fail_angle=0.9):
    """Sum the parent's bounded per-object tilt cost over target and obstacles."""
    if not (0.0 <= free_angle < fail_angle <= math.pi):
        raise ValueError("Tilt thresholds must satisfy 0 <= free_angle < fail_angle <= pi")
    cost = ((all_object_tilt_angles(env) - free_angle) / (fail_angle - free_angle))
    return cost.clamp(0.0, 1.0).square().sum(dim=-1)


def shelf_failure(env):
    """Unit event on the shelf-collision terminal transition, before auto-reset.

    TerminationManager has already evaluated this transition; RewardManager
    runs before reset clears its flags. Cancel its dt multiplier so weight
    -3 means an actual -3 once on the final step, not a -3-per-second rate.
    Timeout or other failures alone do not trigger this event. Auto-reset
    makes each shelf-collision event the last rewarded step of its episode.
    """
    return env.termination_manager.get_term("shelf_collision").float() / env.step_dt


def goal_hold_xy(env, position_tolerance=0.01, command_name="target_goal_pos"):
    """Unit reward rate while the target is inside the world-horizontal disk.

    Only world X/Y are compared: height, shelf tilt, target velocity, and prior
    occupancy do not alter this condition. RewardManager applies weight * dt.
    """
    if not math.isfinite(position_tolerance) or position_tolerance <= 0.0:
        raise ValueError("position_tolerance must be finite and positive")
    objects = env.scene["object_collection"]
    rows = torch.arange(env.num_envs, device=env.device)
    position = objects.data.object_pos_w[rows, env.target_id[:, 0].long()]
    goal = env.command_manager.get_command(command_name)
    distance = torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1)
    return (distance <= position_tolerance).to(position.dtype)


def _current_tactile_bits(env):
    # The manager's copied config contains the resolved, shared term instance;
    # env.cfg still contains its class. ObservationManager has no term getter.
    term = env.observation_manager.cfg.policy.tactile
    return term.func(env, **term.params)


def contact_any(env):
    """Unit rate when at least one of the policy's 17 tactile channels is ON."""
    bits = _current_tactile_bits(env)
    return (bits > 0.5).any(dim=-1).to(bits.dtype)


def contact_count(env, max_contacts=5):
    """Linear channel count, normalized to one at five or more ON channels."""
    if not isinstance(max_contacts, int) or max_contacts <= 0:
        raise ValueError("max_contacts must be a positive integer")
    bits = _current_tactile_bits(env)
    count = (bits > 0.5).sum(dim=-1).to(bits.dtype)
    return count.clamp(max=max_contacts) / max_contacts


def obstacle_displacement_l2(env, max_cost=0.04):
    """Sum squared 3D displacement from exact reset poses, excluding target.

    The cap is in m^2, before the negative reward weight and dt. Extreme or
    non-finite obstacle costs saturate; the target never contributes.
    """
    if not math.isfinite(max_cost) or max_cost <= 0.0:
        raise ValueError("max_cost must be finite and positive")
    position = env.scene["object_collection"].data.object_pos_w
    squared = (position - env._initial_object_positions_w).square().sum(dim=-1)
    is_obstacle = (
        torch.arange(position.shape[1], device=env.device)[None, :]
        != env.target_id.long()
    )
    cost = torch.where(is_obstacle, squared, 0.0).sum(dim=-1)
    return torch.nan_to_num(
        cost, nan=max_cost, posinf=max_cost, neginf=max_cost,
    ).clamp(max=max_cost)


class WrenchChangePenalty(ManagerTermBase):
    """Penalize abrupt changes in the compensated six-axis physical reading.

    Read the same sample as ``policy.wrist_wrench``, expressed in world axes so
    wrist rotation alone does not rotate a steady force into a spurious change.
    The moment remains referenced to the moving EEF observation point.

    Force and moment changes use separate units, deadbands and scales. This is
    a change per control step, not a derivative. RewardManager applies weight
    and dt exactly once. The initial fresh sample only establishes history.

    This term must follow every positive reward term. Their already-updated
    episode sums supply a causal budget: the accumulated new penalty cannot
    exceed ``max_positive_fraction`` of positive reward earned in this episode.
    A separate unit raw cap bounds the absolute cost even on successful runs.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_wrench = torch.zeros((env.num_envs, 6), device=env.device)
        self._previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._term_name = None
        self._positive_names = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_wrench[env_ids] = 0.0
        self._previous_valid[env_ids] = False

    def _reward_budget(self, env, max_positive_fraction):
        """Use current manager sums, without recomputing stateful reward terms."""
        manager = env.reward_manager
        if self._term_name is None:
            names = manager.active_terms
            own_names = [name for name in names if manager.get_term_cfg(name).func is self]
            if len(own_names) != 1:
                raise RuntimeError("WrenchChangePenalty must appear exactly once in RewardManager")
            own_name = own_names[0]
            own_index = names.index(own_name)
            weight = manager.get_term_cfg(own_name).weight
            if not math.isfinite(weight) or weight >= 0.0:
                raise ValueError("WrenchChangePenalty requires a finite negative reward weight")
            positive_names = [
                name for name in names if manager.get_term_cfg(name).weight > 0.0
            ]
            if any(names.index(name) >= own_index for name in positive_names):
                raise RuntimeError("WrenchChangePenalty must follow every positive reward term")
            self._term_name = own_name
            self._positive_names = positive_names
        weight = manager.get_term_cfg(self._term_name).weight
        if not math.isfinite(weight) or weight >= 0.0:
            raise ValueError("WrenchChangePenalty requires a finite negative reward weight")
        earned = torch.zeros(env.num_envs, device=env.device)
        for name in self._positive_names:
            earned += manager._episode_sums[name].clamp_min(0.0)
        # A corrupt positive sum must not create unbounded spending permission.
        earned = torch.nan_to_num(earned, nan=0.0, posinf=0.0, neginf=0.0)
        spent = (-manager._episode_sums[self._term_name]).clamp_min(0.0)
        remaining = (max_positive_fraction * earned - spent).clamp_min(0.0)
        return remaining / (abs(weight) * env.step_dt)

    def __call__(
        self,
        env,
        free_force_change=0.5,
        free_moment_change=0.01,
        force_change_scale=5.0,
        moment_change_scale=0.1,
        max_positive_fraction=0.05,
    ):
        for name, value in (
            ("free_force_change", free_force_change),
            ("free_moment_change", free_moment_change),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("force_change_scale", force_change_scale),
            ("moment_change_scale", moment_change_scale),
            ("step_dt", env.step_dt),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(max_positive_fraction) or not 0.0 <= max_positive_fraction <= 0.05:
            raise ValueError("max_positive_fraction must be in [0, 0.05]")

        term = env.observation_manager.cfg.policy.wrist_wrench
        current = term.func.world_wrench(env)
        valid = env.sensor_data_fresh
        delta = current - self._previous_wrench
        force_excess = (
            torch.linalg.vector_norm(delta[:, :3], dim=-1) - free_force_change
        ).clamp_min(0.0) / force_change_scale
        moment_excess = (
            torch.linalg.vector_norm(delta[:, 3:], dim=-1) - free_moment_change
        ).clamp_min(0.0) / moment_change_scale
        raw_cost = 0.5 * (force_excess.square() + moment_excess.square())
        raw_cost = torch.nan_to_num(raw_cost, nan=1.0, posinf=1.0, neginf=1.0).clamp(max=1.0)
        raw_cost = torch.where(valid & self._previous_valid, raw_cost, 0.0)

        self._previous_wrench[:] = current
        self._previous_valid[:] = valid
        return torch.minimum(raw_cost, self._reward_budget(env, max_positive_fraction))
