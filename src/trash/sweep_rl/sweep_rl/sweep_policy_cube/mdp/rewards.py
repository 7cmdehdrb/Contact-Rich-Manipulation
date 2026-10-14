"""Fixed Sweep-Policy rewards adapted to a single Cube."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .reward_math import (
    axis_alignment_error_kernel,
    ee_y_shelf_z_alignment_kernel,
    fixed_policy_hand_target,
    position_error_tanh_kernel,
    pushing_reward_kernel,
)


def _reaching_position_error(
    env,
    command_name: str,
    target_width: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return EE distance to the Cube-relative Sweep-Policy reaching point."""

    target: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    goal = env.command_manager.get_command(command_name)[:, :3]
    sweep_direction_y = goal[:, 1] - target.data.root_pos_w[:, 1]
    hand_target = fixed_policy_hand_target(
        target.data.root_pos_w,
        target_width,
        sweep_direction_y,
    )
    return torch.linalg.norm(
        hand_target - ee_frame.data.target_pos_w[:, 0], dim=-1
    )


def reaching_position_command_error(
    env,
    command_name: str,
    target_width: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """L2 reaching error, matching Isaac Lab Reach's coarse position term."""

    return _reaching_position_error(
        env, command_name, target_width, object_cfg, ee_frame_cfg
    )


def reaching_position_command_error_tanh(
    env,
    std: float,
    command_name: str,
    target_width: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Fine reaching reward using the Isaac Lab Reach tanh kernel."""

    distance = _reaching_position_error(
        env, command_name, target_width, object_cfg, ee_frame_cfg
    )
    return position_error_tanh_kernel(distance, std)


def reward_for_hand_reaching(
    env,
    command_name: str,
    target_width: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = _reaching_position_error(
        env, command_name, target_width, object_cfg, ee_frame_cfg
    )
    return torch.exp(-0.1 * distance)


def ee_y_shelf_z_orientation_error(
    env,
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the angular error between the EE y-axis and shelf z-axis."""

    shelf: RigidObject = env.scene[shelf_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    shelf_rotation = math_utils.matrix_from_quat(shelf.data.root_quat_w)
    ee_rotation = math_utils.matrix_from_quat(ee_frame.data.target_quat_w[:, 0])
    alignment = ee_y_shelf_z_alignment_kernel(ee_rotation, shelf_rotation)
    return axis_alignment_error_kernel(alignment)


def pushing_target(
    env,
    command_name: str,
    target_width: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    wrist_frame_cfg: SceneEntityCfg = SceneEntityCfg("wrist_frame"),
) -> torch.Tensor:
    target: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    wrist_frame: FrameTransformer = env.scene[wrist_frame_cfg.name]
    goal = env.command_manager.get_command(command_name)[:, :3]
    sweep_direction_y = goal[:, 1] - target.data.root_pos_w[:, 1]
    hand_target = fixed_policy_hand_target(
        target.data.root_pos_w,
        target_width,
        sweep_direction_y,
    )
    distance = torch.linalg.norm(goal - target.data.root_pos_w, dim=-1)
    contact_distance = torch.linalg.norm(
        hand_target - ee_frame.data.target_pos_w[:, 0], dim=-1
    )
    wrist_y_error = torch.abs(
        hand_target[:, 1] - wrist_frame.data.target_pos_w[:, 0, 1]
    )
    return pushing_reward_kernel(
        distance,
        contact_distance,
        wrist_y_error,
        target.data.root_lin_vel_w[:, 1],
    )


def homing_reward(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    goal = env.command_manager.get_command(command_name)[:, :3]
    distance = torch.linalg.norm(goal[:, 1:] - target.data.root_pos_w[:, 1:], dim=-1)
    joint_error = torch.sum(
        torch.abs(robot.data.joint_pos[:, :5] - robot.data.default_joint_pos[:, :5]),
        dim=-1,
    )
    home_score = torch.exp(-0.5 * joint_error)
    return torch.where(distance < 0.03, home_score, 0.0)


def joint_vel_l2_arm(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(robot.data.joint_vel[:, :6].square(), dim=-1)


class ShelfCollisionPenalty(ManagerTermBase):
    """Original shelf movement and link-clearance penalty."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._ee: FrameTransformer = env.scene["ee_frame"]
        self._fingers: FrameTransformer = env.scene["finger_frame"]
        self._wrist: FrameTransformer = env.scene["wrist_frame"]
        self._shelf: RigidObject = env.scene["shelf"]
        self._initial_shelf_pos = (
            self._shelf.data.default_root_state[:, :3] + env.scene.env_origins
        )

    def __call__(self, env) -> torch.Tensor:
        del env
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos
        moved = (
            torch.linalg.norm(shelf_delta, dim=-1)
            + torch.linalg.norm(self._shelf.data.root_vel_w, dim=-1)
            > 0.005
        ).float()
        shelf_surface_z = self._shelf.data.root_pos_w[:, 2] + 1.05
        shelf_reference = self._shelf.data.root_pos_w.clone()
        shelf_reference[:, 2] = shelf_surface_z
        near_shelf = (
            torch.linalg.norm(
                shelf_reference - self._ee.data.target_pos_w[:, 0], dim=-1
            )
            < 0.2
        ).float()
        left_clearance = self._fingers.data.target_pos_w[:, 0, 2] - shelf_surface_z
        right_clearance = self._fingers.data.target_pos_w[:, 1, 2] - shelf_surface_z
        wrist_clearance = self._wrist.data.target_pos_w[:, 0, 2] - shelf_surface_z
        clearance_penalty = (
            torch.clamp(1.0 - left_clearance / 0.02, 0.0, 1.0)
            + torch.clamp(1.0 - right_clearance / 0.02, 0.0, 1.0)
            + torch.clamp(1.0 - wrist_clearance / 0.08, 0.0, 1.0)
        )
        return moved + near_shelf * clearance_penalty
