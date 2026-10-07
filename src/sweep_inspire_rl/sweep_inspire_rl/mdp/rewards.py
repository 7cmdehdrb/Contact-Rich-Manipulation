"""Sweep-Policy rewards adapted to a single palmar pushing surface."""

from __future__ import annotations

import torch
from isaaclab.utils.math import matrix_from_quat
from sweeping_policy.mdp.reward_random_sweep import pushing_target as source_pushing_target

from .events import collision_aabbs
from .observations import palm_tactile_bits


PALM_SURFACE_POINT_H = (0.0004, 0.0127, 0.1196)
CONTROL_POINT_OFFSET_H = (0.0, 0.05, 0.10)


def palm_surface_position(env):
    """World position of the physical pad's front surface, rather than C."""
    frame = env.scene["ee_frame"].data
    rotation = matrix_from_quat(frame.target_quat_w[:, 0])
    offset = frame.target_pos_w.new_tensor(PALM_SURFACE_POINT_H) - frame.target_pos_w.new_tensor(CONTROL_POINT_OFFSET_H)
    return frame.target_pos_w[:, 0] + (rotation @ offset.expand(env.num_envs, -1).unsqueeze(-1)).squeeze(-1)


def upstream_surface_position(env):
    """Use the selected USD collider's yaw-aware upstream bounding surface."""
    state = env.scene["object_collection"].data.object_link_state_w[:, 0]
    minimum, _ = collision_aabbs(
        state[:, :3], state[:, 3:7],
        env._target_collision_local_center, env._target_collision_half_extent,
    )
    return minimum[:, 1]


def reaching_position(env, z_offset=0.12):
    objects = env.scene["object_collection"]
    target = objects.data.object_pos_w[:, 0].clone()
    target[:, 1] = upstream_surface_position(env)
    target[:, 2] += z_offset
    return target


def hand_reaching(env):
    """Keep Sweep's exp(-10*d) reaching kernel at the raised palm pose."""
    hand = palm_surface_position(env)
    return torch.exp(-10.0 * torch.linalg.vector_norm(reaching_position(env) - hand, dim=-1))


def palm_alignment(env):
    """Signed palmar-normal/right and hand-up/world-up alignment."""
    rotation = matrix_from_quat(env.scene["ee_frame"].data.target_quat_w[:, 0])
    right = rotation[:, 1, 1]
    up = rotation[:, 2, 0]
    return 0.5 * (right * right.abs() + up * up.abs())


def pushing_target(env, command_name="target_goal_pos", eef_distance_threshold=0.052):
    """Sweep-Policy reward with a 5.2 cm EEF gate; sensors are observations only."""
    result = source_pushing_target(
        env, command_name=command_name, eef_distance_threshold=eef_distance_threshold
    )
    diagnostic_env = getattr(env, "sweep_gate_diagnostic_env", None)
    if diagnostic_env is None:
        return result

    # Mirror only the source gate for logging. The reward above is evaluated
    # by the source function itself, including its ungated goal-region branch.
    objects = env.scene["object_collection"]
    rows = torch.arange(env.num_envs, device=env.target_id.device)
    ids = env.target_id.squeeze(-1).long()
    target = objects.data.object_pos_w[rows, ids]
    goal = env.command_manager.get_command(command_name)[:, :3]
    offset = target.clone()
    offset[:, 0] -= 0.02
    offset[:, 1] -= env.target_width[:, 0] * torch.sign(env.sweep_dir[:, 1])
    offset[:, 2] += 0.09
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    wrist = env.scene["wrist_frame"].data.target_pos_w[:, 0]
    ee_distance = torch.norm(offset - ee, dim=-1, p=2)
    wrist_y_distance = torch.abs(offset[:, 1] - wrist[:, 1])
    near_hand = ee_distance < eef_distance_threshold
    near_wrist = wrist_y_distance < 0.04
    distance = torch.norm(goal - target, dim=-1, p=2)
    tactile = palm_tactile_bits(env)
    hand = palm_surface_position(env)
    values = {
        "episode_step": env.episode_length_buf,
        "reaching_distance_m": ee_distance,
        "wrist_y_distance_m": wrist_y_distance,
        "near_hand": near_hand,
        "near_wrist": near_wrist,
        "gate": near_hand & near_wrist,
        "goal_region": distance < 0.03,
        "goal_distance_m": distance,
        "object_velocity_y_m_s": objects.data.object_lin_vel_w[rows, ids, 1],
        "palm_contact": tactile[:, 0] > 0.5,
        "other_pad_contact": (tactile[:, 1:] > 0.5).any(dim=1),
        "alignment": palm_alignment(env),
        "sweeping_raw": result,
    }
    sensor = env.scene["palm_tactile"]
    pad_index = sensor.body_names.index("inspire_palm_force_sensor")
    values["palm_force_n"] = torch.linalg.vector_norm(sensor.data.net_forces_w[:, pad_index], dim=-1)
    if hasattr(env, "sensor_data_fresh"):
        values["sensor_data_fresh"] = env.sensor_data_fresh
    for axis, name in enumerate(("x", "y", "z")):
        values[f"eef_{name}_w_m"] = ee[:, axis]
        values[f"palm_{name}_w_m"] = hand[:, axis]
        values[f"object_{name}_w_m"] = target[:, axis]
    values.update({
        f"done_{name}": env.termination_manager.get_term(name)
        for name in env.termination_manager.active_terms
    })
    # Copy before automatic reset, including the terminal step's state.
    env.sweep_gate_diagnostics = {
        name: value[diagnostic_env].detach().clone() for name, value in values.items()
    }
    return result


def hand_velocity_limit(env, threshold=1.0):
    """Resolve arm joints by name instead of the old first-six ordering."""
    term = env.action_manager.get_term("arm_action")
    return (env.scene["robot"].data.joint_vel[:, term._joint_ids].abs() > threshold).any(dim=1)
