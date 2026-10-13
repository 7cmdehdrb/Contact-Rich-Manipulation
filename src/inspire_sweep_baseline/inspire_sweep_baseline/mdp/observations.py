"""The 34D sweep observation prefix plus palm contact 1D and F/T 6D."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_unique, subtract_frame_transforms

from .sensors import FixedJointWrenchReader, contact_magnitudes_and_bits


def arm_joint_pos_rel(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]


def arm_joint_vel_rel(env, asset_cfg=SceneEntityCfg("robot")):
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids] - robot.data.default_joint_vel[:, asset_cfg.joint_ids]


def last_action(env):
    return env.action_manager.action


def object_position_in_root(env):
    robot = env.scene["robot"]
    objects = env.scene["object_collection"]
    rows = torch.arange(env.num_envs, device=env.device)
    positions = objects.data.object_pos_w[rows, env.target_id[:, 0].long()]
    position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, positions)
    return position


def object_width(env):
    return env.target_width


def eef_pose_in_root(env):
    robot = env.scene["robot"]
    frame = env.scene["ee_frame"].data
    position, rotation = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        frame.target_pos_w[:, 0], frame.target_quat_w[:, 0],
    )
    return torch.cat((position, quat_unique(rotation)), dim=-1)


def goal_position_in_root(env, command_name="target_goal_pos"):
    robot = env.scene["robot"]
    goal = env.command_manager.get_command(command_name)[:, :3]
    position, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal)
    return position


def palm_contact_bit(env, sensor_name="palm_tactile", threshold_n=0.05):
    """One actual palm pad only, thresholded at 0.05 N by default."""
    fresh = env.sensor_data_fresh
    if not bool(fresh.any()):
        return torch.zeros((env.num_envs, 1), device=env.device)
    sensor = env.scene[sensor_name]
    if tuple(sensor.body_names) != ("inspire_palm_force_sensor",):
        raise RuntimeError(f"Expected only the palm pad, got {sensor.body_names}")
    forces = torch.where(fresh[:, None, None], sensor.data.net_forces_w, torch.zeros_like(sensor.data.net_forces_w))
    _, bits = contact_magnitudes_and_bits(forces, threshold_n)
    return bits.to(torch.float32)


def wrist_wrench_eef(env, asset_name="robot", frame_name="ee_frame"):
    """[Fx,Fy,Fz,Mx,My,Mz], uncancelled N/Nm, expressed/referenced at EEF."""
    fresh = env.sensor_data_fresh
    if not bool(fresh.any()):
        return torch.zeros((env.num_envs, 6), device=env.device)
    robot = env.scene[asset_name]
    cache = getattr(env, "_baseline_wrench_readers", None)
    if cache is None:
        cache = env._baseline_wrench_readers = {}
    if asset_name not in cache:
        names = tuple(robot.body_names)
        if names.count("axia80_link") != 1:
            raise RuntimeError("Expected exactly one Axia80 body")
        cache[asset_name] = (FixedJointWrenchReader(robot), names.index("axia80_link"))
    reader, f_body_id = cache[asset_name]
    frame = env.scene[frame_name].data
    c_pos_w, c_quat_w = frame.target_pos_w[:, 0], frame.target_quat_w[:, 0]
    f_pos_w, f_quat_w = robot.data.body_pos_w[:, f_body_id], robot.data.body_quat_w[:, f_body_id]
    rotation = matrix_from_quat(c_quat_w).transpose(-1, -2) @ matrix_from_quat(f_quat_w)
    moment_arm = quat_apply_inverse(c_quat_w, f_pos_w - c_pos_w)
    result = reader.read(rotation, moment_arm, valid_env_mask=fresh).measured_c
    return torch.where(fresh[:, None], result, torch.zeros_like(result))
