"""Reset-safe wrench-change cost with an episode-relative reward budget."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase


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
