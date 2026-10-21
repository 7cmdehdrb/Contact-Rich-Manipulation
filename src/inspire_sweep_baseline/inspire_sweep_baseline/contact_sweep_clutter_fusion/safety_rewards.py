"""Independent, bounded EEF and compensated-wrench costs for Clutter Fusion."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from ..mdp.goal_rewards import _exponential_speed_excess_limit


def eef_linear_speed_cost(
    env, free_speed=1.5, max_cost=100.0,
    asset_name="robot", body_name="axia80_link", frame_name="ee_frame",
):
    """Bounded exponential speed cost at the actual EEF point, in m/s.

    Link-origin velocity and angular velocity are instantaneous simulation
    quantities. Transport the velocity to the FrameTransformer's target point;
    neither COM velocity nor a control-step position difference is substituted.
    RewardManager applies the negative weight and control dt once.
    """
    robot = env.scene[asset_name]
    body_index = robot.body_names.index(body_name)
    data = robot.data
    offset_w = env.scene[frame_name].data.target_pos_w[:, 0] - data.body_link_pos_w[:, body_index]
    velocity_w = data.body_link_lin_vel_w[:, body_index] + torch.cross(
        data.body_link_ang_vel_w[:, body_index], offset_w, dim=-1,
    )
    speed = torch.linalg.vector_norm(velocity_w, dim=-1)
    speed = torch.nan_to_num(speed, nan=math.inf, posinf=math.inf, neginf=math.inf)
    return _exponential_speed_excess_limit(speed, free_speed, max_cost)


def eef_upward_z_orientation_cost(env, onset_angle=math.pi / 3.0, frame_name="ee_frame"):
    """Finite [0, 1] cost when EEF +Z enters the upward cone.

    The EEF +Z direction points away from the palm normal. Upward +Z therefore
    means a downward-facing palm. Only angles smaller than ``onset_angle`` to
    world +Z are penalized; a downward +Z (upward-facing palm) is free.
    A fixed reward weight provides the calibrated scale, with no reward budget.
    """
    if not math.isfinite(onset_angle) or not 0.0 < onset_angle <= math.pi / 2.0:
        raise ValueError("onset_angle must be finite and in (0, pi/2]")
    quaternion = env.scene[frame_name].data.target_quat_w[:, 0]
    local_z = torch.zeros_like(quaternion[:, :3])
    local_z[:, 2] = 1.0
    cosine = quat_apply(quaternion, local_z)[:, 2]
    cosine = torch.nan_to_num(cosine, nan=1.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    threshold = math.cos(onset_angle)
    return ((cosine - threshold) / (1.0 - threshold)).clamp(0.0, 1.0).square()


class WrenchRatePenalty(ManagerTermBase):
    """Bounded exponential cost of unusually large compensated wrench rates.

    Force and moment use separate physical units. World axes avoid a spurious
    delta when only the EEF axes rotate; moments retain the moving EEF origin.
    The maximum normalized force/moment excess defines one combined severity.
    Both the exponent and raw result are bounded before weighting.

    The term must follow all positive reward terms. At weight -0.1 its absolute
    maximum is -0.1/s, while cumulative cost cannot exceed 5% of positive reward
    already earned in the current episode. Reset/invalid history is never used
    for a rate. Repeated calls in the same physical step reuse the raw result.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_wrench = torch.zeros((env.num_envs, 6), device=env.device)
        self._previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._cached_raw = torch.zeros(env.num_envs, device=env.device)
        self._cached_key = None
        self._term_name = None
        self._positive_names = None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._previous_wrench[env_ids] = 0.0
        self._previous_valid[env_ids] = False
        self._cached_raw[env_ids] = 0.0
        self._cached_key = None

    def _reward_budget(self, env, max_positive_fraction):
        manager = env.reward_manager
        if self._term_name is None:
            names = manager.active_terms
            own_names = [name for name in names if manager.get_term_cfg(name).func is self]
            if len(own_names) != 1:
                raise RuntimeError("WrenchRatePenalty must appear exactly once in RewardManager")
            own_name = own_names[0]
            weight = manager.get_term_cfg(own_name).weight
            if not math.isfinite(weight) or weight >= 0.0:
                raise ValueError("WrenchRatePenalty requires a finite negative reward weight")
            positive_names = [name for name in names if manager.get_term_cfg(name).weight > 0.0]
            if any(names.index(name) >= names.index(own_name) for name in positive_names):
                raise RuntimeError("WrenchRatePenalty must follow every positive reward term")
            self._term_name = own_name
            self._positive_names = positive_names
        weight = manager.get_term_cfg(self._term_name).weight
        if not math.isfinite(weight) or weight >= 0.0:
            raise ValueError("WrenchRatePenalty requires a finite negative reward weight")
        earned = torch.zeros(env.num_envs, device=env.device)
        for name in self._positive_names:
            earned += manager._episode_sums[name].clamp_min(0.0)
        earned = torch.nan_to_num(earned, nan=0.0, posinf=0.0, neginf=0.0)
        spent = (-manager._episode_sums[self._term_name]).clamp_min(0.0)
        spent = torch.nan_to_num(spent, nan=math.inf, posinf=math.inf, neginf=0.0)
        remaining = (max_positive_fraction * earned - spent).clamp_min(0.0)
        return remaining / (abs(weight) * env.step_dt)

    def __call__(
        self, env, free_force_rate=2000.0, free_moment_rate=100.0,
        saturation_force_rate=5000.0, saturation_moment_rate=250.0,
        sharpness=3.0, max_positive_fraction=0.05,
    ):
        for name, free, saturation in (
            ("force", free_force_rate, saturation_force_rate),
            ("moment", free_moment_rate, saturation_moment_rate),
        ):
            if not math.isfinite(free) or free < 0.0:
                raise ValueError(f"free_{name}_rate must be finite and nonnegative")
            if not math.isfinite(saturation) or saturation <= free:
                raise ValueError(f"saturation_{name}_rate must be finite and exceed free rate")
        if not math.isfinite(sharpness) or not 0.0 < sharpness <= 50.0:
            raise ValueError("sharpness must be finite and in (0, 50]")
        if not math.isfinite(env.step_dt) or env.step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")
        if not math.isfinite(max_positive_fraction) or not 0.0 <= max_positive_fraction <= 0.05:
            raise ValueError("max_positive_fraction must be in [0, 0.05]")

        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_key:
            reader = env.observation_manager.cfg.policy.wrist_wrench.func
            current = reader.world_wrench(env)
            fresh = env.sensor_data_fresh
            finite = torch.isfinite(current).all(dim=-1)
            delta = current - self._previous_wrench
            force_rate = torch.linalg.vector_norm(delta[:, :3], dim=-1) / env.step_dt
            moment_rate = torch.linalg.vector_norm(delta[:, 3:], dim=-1) / env.step_dt
            force_excess = (force_rate - free_force_rate).clamp_min(0.0) / (saturation_force_rate - free_force_rate)
            moment_excess = (moment_rate - free_moment_rate).clamp_min(0.0) / (saturation_moment_rate - free_moment_rate)
            severity = torch.maximum(force_excess, moment_excess)
            severity = torch.nan_to_num(severity, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            raw = (torch.expm1(sharpness * severity) / math.expm1(sharpness)).clamp(0.0, 1.0)
            self._cached_raw[:] = torch.where(fresh & self._previous_valid, raw, 0.0)
            self._previous_wrench[:] = torch.where((fresh & finite)[:, None], current, 0.0)
            self._previous_valid[:] = fresh & finite
            self._cached_key = key
        return torch.minimum(self._cached_raw, self._reward_budget(env, max_positive_fraction))
