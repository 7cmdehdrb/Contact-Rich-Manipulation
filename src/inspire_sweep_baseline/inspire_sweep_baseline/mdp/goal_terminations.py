"""Failure and sustained object-goal success rules for the omni sweep task."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms

from .goal_rewards import object_tilt_angle


def _object_position(env):
    rows = torch.arange(env.num_envs, device=env.device)
    return env.scene["object_collection"].data.object_pos_w[rows, env.target_id[:, 0].long()]


def _object_position_in_shelf(env):
    shelf = env.scene["shelf"].data
    position, _ = subtract_frame_transforms(shelf.root_pos_w, shelf.root_quat_w, _object_position(env))
    return position


def object_fall_or_tilt(env, height_condition=1.04, fail_angle=0.9):
    """Keep the baseline drop height; use the angle to the shelf normal for tilt."""
    height = _object_position(env)[:, 2] - env.scene.env_origins[:, 2]
    return (height < height_condition) | (object_tilt_angle(env) >= fail_angle)


def object_outside_shelf(env, half_extents=(0.20, 0.50)):
    """Terminate when the object's bottom-origin center leaves the shelf board."""
    position = _object_position_in_shelf(env)
    bounds = position.new_tensor(half_extents)
    return torch.any(position[:, :2].abs() > bounds, dim=-1)


def shelf_displacement(env, threshold=0.05):
    shelf = env.scene["shelf"].data
    initial = shelf.default_root_state[:, :3] + env.scene.env_origins
    return torch.linalg.vector_norm(shelf.root_pos_w - initial, dim=-1) > threshold


class StableGoalSuccess(ManagerTermBase):
    """Require a supported, upright, slow object at its goal for a hold interval.

    This term must follow the named failure terms in the termination config.
    Rewards read its result after the termination manager computes all terms.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._hold_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        self._hold_steps[slice(None) if env_ids is None else env_ids] = 0

    @property
    def hold_duration_s(self):
        """Current consecutive eligible time, updated once by terminations.

        Rewards may read this after termination computation without advancing
        the counter a second time. Partial resets clear only the reset rows.
        """
        return self._hold_steps * self._env.step_dt

    def __call__(
        self, env, command_name="target_goal_pos", position_tolerance=0.01,
        speed_tolerance=0.02, upright_angle=math.radians(10.0), hold_time=0.3,
        surface_height=1.05, support_tolerance=0.01, half_extents=(0.20, 0.50),
        object_radius=0.04, failure_terms=(),
    ):
        if hold_time < 0.0 or support_tolerance <= 0.0 or object_radius <= 0.0:
            raise ValueError("hold_time must be nonnegative and support tolerance/radius positive")
        position = _object_position(env)
        goal = env.command_manager.get_command(command_name)[:, :3]
        rows = torch.arange(env.num_envs, device=env.device)
        velocity = env.scene["object_collection"].data.object_lin_vel_w[rows, env.target_id[:, 0].long()]
        local_position = _object_position_in_shelf(env)
        tilt = object_tilt_angle(env)
        # The cylinder's link origin is the center of its bottom face. Its
        # lowest rim is radius*sin(tilt) below that origin when nearly upright.
        bottom_clearance = local_position[:, 2] - object_radius * torch.sin(tilt) - surface_height
        bounds = position.new_tensor(half_extents) - object_radius
        supported = (
            torch.all(local_position[:, :2].abs() <= bounds, dim=-1)
            & (bottom_clearance.abs() <= support_tolerance)
        )
        eligible = (
            (torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1) < position_tolerance)
            & (torch.linalg.vector_norm(velocity[:, :2], dim=-1) < speed_tolerance)
            & (tilt <= upright_angle)
            & supported
        )
        for name in failure_terms:
            eligible &= ~env.termination_manager.get_term(name)
        self._hold_steps[:] = torch.where(eligible, self._hold_steps + 1, 0)
        required_steps = max(1, math.ceil(hold_time / env.step_dt - 1.0e-9))
        return self._hold_steps >= required_steps
