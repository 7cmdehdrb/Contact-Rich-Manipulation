"""Failure conditions for the fixed-direction sweep task."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .task_math import (
    contact_force_mask,
    object_drop_mask,
    speed_limit_mask,
    target_reached_and_stopped_mask,
    update_sustained_violation,
)


def robot_shelf_collision(
    env,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    """Terminate when any robot body produces filtered contact with the shelf."""
    force_matrix_w = env.scene[sensor_name].data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' has no filtered force matrix."
        )
    return contact_force_mask(force_matrix_w, force_threshold=force_threshold)


def object_drop(
    env,
    command_name: str,
    maximum_height_change: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    return object_drop_mask(
        target.data.root_pos_w[:, 2],
        command.initial_pos_w[:, 2],
        maximum_height_change=maximum_height_change,
    )


def target_reached_and_stopped(
    env,
    command_name: str,
    endpoint_threshold: float,
    speed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate successfully once the target reaches the endpoint and stops."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    goal_distance = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    object_speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return target_reached_and_stopped_mask(
        goal_distance,
        object_speed,
        distance_threshold=endpoint_threshold,
        speed_threshold=speed_threshold,
    )


class PushFastTermination(ManagerTermBase):
    """Terminate sustained overspeed with a 0.30 -> 0.20 -> 0.15 m/s curriculum."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_steps[env_ids] = 0

    def __call__(
        self,
        env,
        initial_speed: float,
        intermediate_speed: float,
        final_speed: float,
        first_transition_step: int,
        second_transition_step: int,
        sustained_steps: int,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ) -> torch.Tensor:
        if not (
            initial_speed >= intermediate_speed >= final_speed > 0.0
            and 0 <= first_transition_step < second_transition_step
        ):
            raise ValueError("Invalid push-fast curriculum configuration.")
        if env.common_step_counter < first_transition_step:
            maximum_speed = initial_speed
        elif env.common_step_counter < second_transition_step:
            maximum_speed = intermediate_speed
        else:
            maximum_speed = final_speed
        target: RigidObject = env.scene[object_cfg.name]
        violation = speed_limit_mask(
            target.data.root_lin_vel_w, maximum_speed=maximum_speed
        )
        self._consecutive_steps, terminated = update_sustained_violation(
            self._consecutive_steps,
            violation,
            required_steps=sustained_steps,
        )
        return terminated
