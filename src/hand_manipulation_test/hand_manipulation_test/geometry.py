"""Shared control-frame and physical sensor access for manager terms."""

from __future__ import annotations

import torch

from .action_math import (
    compose_fixed_offset_pose,
    quaternion_conjugate,
    quaternion_rotate,
    quaternion_to_matrix,
)
from .sensors import FixedJointWrenchReader, FixedJointWrenchSample, PalmTactileReader


def _robot(env):
    return env.scene[env.cfg.actions.arm_action.asset_name]


def _body_index(env, body_name: str) -> int:
    cache = getattr(env, "_reaching_body_indices", None)
    if cache is None:
        cache = env._reaching_body_indices = {}
    key = (env.cfg.actions.arm_action.asset_name, body_name)
    if key not in cache:
        indices, names = _robot(env).find_bodies(body_name, preserve_order=True)
        if len(indices) != 1 or tuple(names) != (body_name,):
            raise RuntimeError(f"Expected one robot body {body_name!r}, got {names}")
        cache[key] = int(indices[0])
    return cache[key]


def base_link_pose_w(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the actual robot base_link origin and axes in world coordinates."""

    robot = _robot(env)
    body_index = _body_index(env, "base_link")
    return robot.data.body_link_pos_w[:, body_index], robot.data.body_link_quat_w[:, body_index]


def control_point_pose_w(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the massless C/EEF frame from the configured rigid hand offset."""

    robot = _robot(env)
    cfg = env.cfg.actions.arm_action
    body_index = _body_index(env, cfg.body_name)
    position_h = robot.data.body_link_pos_w[:, body_index]
    quaternion_h = robot.data.body_link_quat_w[:, body_index]
    offset = position_h.new_tensor(cfg.body_offset_pos).expand(position_h.shape[0], -1)
    offset_quaternion = quaternion_h.new_tensor(cfg.body_offset_quat).expand(quaternion_h.shape[0], -1)
    position_c, quaternion_c, _ = compose_fixed_offset_pose(
        position_h, quaternion_h, offset, offset_quaternion
    )
    return position_c, quaternion_c


def wrist_wrench_c(env) -> FixedJointWrenchSample:
    """Rotate the raw F/T reaction into C and shift its moment reference."""

    robot = _robot(env)
    reader = getattr(env, "_reaching_wrench_reader", None)
    if reader is None:
        reader = env._reaching_wrench_reader = FixedJointWrenchReader(robot, "inspire_base_link")
    sensor_index = _body_index(env, "axia80_link")
    position_f = robot.data.body_link_pos_w[:, sensor_index]
    quaternion_f = robot.data.body_link_quat_w[:, sensor_index]
    position_c, quaternion_c = control_point_pose_w(env)
    r_c_to_f_c = quaternion_rotate(quaternion_conjugate(quaternion_c), position_f - position_c)
    rotation_c_from_f = quaternion_to_matrix(quaternion_c).transpose(-1, -2) @ quaternion_to_matrix(quaternion_f)
    return reader.read(rotation_c_from_f, r_c_to_f_c)


def palm_tactile_bits(env) -> torch.Tensor:
    """Return seventeen physical palmar-pad bits in the canonical order."""

    reader = getattr(env, "_reaching_palm_tactile_reader", None)
    if reader is None:
        reader = env._reaching_palm_tactile_reader = PalmTactileReader(
            env.scene["palm_tactile"], threshold_n=env.cfg.task.tactile_threshold_n
        )
    return reader.read().bits.to(dtype=torch.float32)
