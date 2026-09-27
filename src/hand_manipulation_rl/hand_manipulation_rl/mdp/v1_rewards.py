"""Corrective rewards for the inherited Approach-v1 environment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase

from .approach_rewards import (
    one_sided_height_safety_penalty,
    potential_progress_rate,
    selected_surface_contact_path,
)
from .rewards import target_gated_tactile_contact_reward


class SelectedSurfaceApproachProgress(ManagerTermBase):
    """Signed, telescoping progress along the contact-and-push path."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._reference_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._previous_potential[index] = 0.0
        self._reference_valid[index] = False

    def capture_reference(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        """Capture the actual collision-certified pose after reset finalization."""

        index = slice(None) if env_ids is None else env_ids
        potential = selected_surface_contact_path(self._env).detach()
        self._previous_potential[index] = potential[index]
        self._reference_valid[index] = True

    def __call__(self, env) -> torch.Tensor:
        if env is not self._env:
            raise RuntimeError("Approach progress term was called with a different environment")
        current = selected_surface_contact_path(env)
        value = potential_progress_rate(
            current,
            self._previous_potential,
            env.step_dt,
        )
        value = torch.where(self._reference_valid, value, torch.zeros_like(value))
        self._previous_potential[:] = current.detach()
        self._reference_valid[:] = True
        return value


def selected_surface_target_tactile_contact(env) -> torch.Tensor:
    """Positive tactile reward gated by selected-surface TargetObject contact."""

    sensor_live = env.sensor_valid & env.sensor_data_fresh
    bits = env.selected_surface_object_tactile_bits()
    bits = torch.where(sensor_live.unsqueeze(-1), bits, torch.zeros_like(bits))
    target_contact = torch.any(bits.bool(), dim=-1) & sensor_live
    return target_gated_tactile_contact_reward(
        bits,
        target_contact,
        beta=env.cfg.task.tactile_contact_beta,
    )


class TargetHandObjectContactAcquisition(ManagerTermBase):
    """One-shot bonus for first meaningful Hand--Cube contact per episode."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._acquired = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        index = slice(None) if env_ids is None else env_ids
        self._acquired[index] = False

    def __call__(self, env) -> torch.Tensor:
        if env is not self._env:
            raise RuntimeError("Contact acquisition term was called with a different environment")
        sensor_live = env.sensor_valid & env.sensor_data_fresh
        contact = env.hand_object_contact() & sensor_live
        acquired_now = contact & ~self._acquired
        self._acquired |= contact
        # RewardManager integrates by step_dt. Divide once so cfg.weight is
        # the complete episode-level acquisition bonus, not a per-step rate.
        return acquired_now.to(dtype=torch.float32) / env.step_dt


def selected_surface_height_safety(env) -> torch.Tensor:
    """Penalize descent below the calibrated contact path, before the shelf."""

    current_height = env.selected_surface_control_point_w()[:, 2]
    contact_path_height = env.selected_surface_approach_height_w()
    return one_sided_height_safety_penalty(
        current_height,
        contact_path_height,
        tolerance_m=env.cfg.task.safe_height_below_contact_path_m,
        band_m=env.cfg.task.safe_height_penalty_band_m,
    )


__all__ = [
    "SelectedSurfaceApproachProgress",
    "TargetHandObjectContactAcquisition",
    "selected_surface_height_safety",
    "selected_surface_target_tactile_contact",
]
