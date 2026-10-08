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


def hand_reaching_object_center(env, z_offset=0.075):
    """V1/V3: follow live object XYZ with a raised EEF target."""
    target = env.scene["object_collection"].data.object_pos_w[:, 0].clone()
    target[:, 2] += z_offset
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    return torch.exp(-10.0 * torch.linalg.vector_norm(target - ee, dim=-1))


def hand_reaching_fixed_height(env, z_offset=0.075, command_name="target_goal_pos"):
    """V2: use Sweep-Policy's gate XY offset with episode-fixed target Z."""
    target = env.scene["object_collection"].data.object_pos_w[:, 0].clone()
    target[:, 0] -= 0.02
    target[:, 1] -= env.target_width[:, 0] * torch.sign(env.sweep_dir[:, 1])
    target[:, 2] = env.command_manager.get_command(command_name)[:, 2] + z_offset
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    return torch.exp(-10.0 * torch.linalg.vector_norm(target - ee, dim=-1))


def sweeping_height_error(env, z_offset=0.075, height_scale=0.015,
                          eef_distance_threshold=0.04, command_name="target_goal_pos",
                          wrist_y_offset=0.0):
    """Penalize height drift during planar sweeping, without adding a Z gate.

    The command Z is latched at reset, so tipping/lifting the cup cannot raise
    the target height. One height_scale of error returns one penalty unit.
    """
    state = _sweep_gate_state(env, command_name, eef_distance_threshold, True, z_offset, wrist_y_offset)
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    desired_z = env.command_manager.get_command(command_name)[:, 2] + z_offset
    return state["gate"].to(ee.dtype) * torch.square((ee[:, 2] - desired_z) / height_scale)


def palm_alignment(env):
    """Signed palmar-normal/right and hand-up/world-up alignment."""
    rotation = matrix_from_quat(env.scene["ee_frame"].data.target_quat_w[:, 0])
    right = rotation[:, 1, 1]
    up = rotation[:, 2, 0]
    return 0.5 * (right * right.abs() + up * up.abs())


def hand_up_alignment(env):
    """Match the source signed-square kernel using Inspire +X and shelf +Z."""
    hand_rotation = matrix_from_quat(env.scene["ee_frame"].data.target_quat_w[:, 0])
    shelf_rotation = matrix_from_quat(env.scene["shelf"].data.default_root_state[:, 3:7])
    alignment = torch.sum(hand_rotation[:, :, 0] * shelf_rotation[:, :, 2], dim=-1)
    return alignment * alignment.abs()


def _sweep_gate_state(env, command_name, eef_distance_threshold, eef_distance_xy_only,
                      pushing_z_offset, wrist_y_offset):
    """Shared gate geometry for reward, height penalty and diagnostics."""
    objects = env.scene["object_collection"]
    rows = torch.arange(env.num_envs, device=env.target_id.device)
    ids = env.target_id.squeeze(-1).long()
    target = objects.data.object_pos_w[rows, ids]
    goal = env.command_manager.get_command(command_name)[:, :3]
    offset = target.clone()
    offset[:, 0] -= 0.02
    offset[:, 1] -= env.target_width[:, 0] * torch.sign(env.sweep_dir[:, 1])
    offset[:, 2] += pushing_z_offset
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    wrist = env.scene["wrist_frame"].data.target_pos_w[:, 0]
    ee_delta = offset - ee
    if eef_distance_xy_only:
        ee_delta = ee_delta[:, :2]
    ee_distance = torch.norm(ee_delta, dim=-1, p=2)
    desired_wrist_y = offset[:, 1] + wrist_y_offset
    wrist_y_distance = torch.abs(desired_wrist_y - wrist[:, 1])
    near_hand = ee_distance < eef_distance_threshold
    near_wrist = wrist_y_distance < 0.04
    distance = torch.norm(goal - target, dim=-1, p=2)
    return {
        "target": target, "goal": goal, "rows": rows, "ids": ids,
        "reaching_distance_m": ee_distance,
        "wrist_y_distance_m": wrist_y_distance,
        "wrist_y_distance_uncompensated_m": torch.abs(offset[:, 1] - wrist[:, 1]),
        "desired_wrist_y_w_m": desired_wrist_y,
        "eef_wrist_y_separation_m": ee[:, 1] - wrist[:, 1],
        "near_hand": near_hand, "near_wrist": near_wrist,
        "gate": near_hand & near_wrist,
        "goal_region": distance < 0.03, "goal_distance_m": distance,
    }


def pushing_target(env, command_name="target_goal_pos", eef_distance_threshold=0.09,
                   eef_distance_xy_only=False, pushing_z_offset=0.09,
                   height_reference_initial=False, wrist_y_offset=0.0):
    """Keep source shaping; V2 compensates the wrist gate's geometric offset."""
    diagnostic_env = getattr(env, "sweep_gate_diagnostic_env", None)
    record_metrics = getattr(env, "_record_sweep_metrics", None)
    if wrist_y_offset == 0.0:
        result = source_pushing_target(
            env, command_name=command_name, eef_distance_threshold=eef_distance_threshold,
            eef_distance_xy_only=eef_distance_xy_only, pushing_z_offset=pushing_z_offset,
        )
        if diagnostic_env is None and record_metrics is None:
            return result
    state = _sweep_gate_state(env, command_name, eef_distance_threshold,
                              eef_distance_xy_only, pushing_z_offset, wrist_y_offset)
    objects = env.scene["object_collection"]
    rows, ids = state["rows"], state["ids"]
    target, goal = state["target"], state["goal"]
    velocity_y = objects.data.object_lin_vel_w[rows, ids, 1]
    if wrist_y_offset != 0.0:
        # Identical source distance/velocity/goal-region shaping; only gate changes.
        speed = velocity_y.abs()
        velocity_reward = torch.where(speed > 0.05, torch.where(speed < 0.1, 0.5, -0.5), 0.0)
        distance = state["goal_distance_m"]
        result = torch.where(
            state["goal_region"], 2.0 * torch.exp(-5.0 * distance),
            state["gate"] * ((1.0 - distance / 0.18) + velocity_reward),
        )
    if diagnostic_env is None and record_metrics is None:
        return result
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0]
    tactile = palm_tactile_bits(env)
    hand = palm_surface_position(env)
    desired_z = (goal[:, 2] if height_reference_initial else target[:, 2]) + pushing_z_offset
    orientation_term = env.reward_manager.get_term_cfg("orientation") if hasattr(env, "reward_manager") else None
    alignment = (orientation_term.func(env, **orientation_term.params)
                 if orientation_term is not None else palm_alignment(env))
    values = {
        **{key: value for key, value in state.items() if key not in ("target", "goal", "rows", "ids")},
        "episode_step": env.episode_length_buf,
        "object_velocity_y_m_s": velocity_y,
        "palm_contact": tactile[:, 0] > 0.5,
        "other_pad_contact": (tactile[:, 1:] > 0.5).any(dim=1),
        "alignment": alignment,
        "sweeping_raw": result,
        "desired_eef_z_w_m": desired_z,
        "eef_height_error_m": ee[:, 2] - desired_z,
    }
    sensor = env.scene["palm_tactile"]
    pad_index = sensor.body_names.index("inspire_palm_force_sensor")
    values["palm_force_n"] = torch.linalg.vector_norm(sensor.data.net_forces_w[:, pad_index], dim=-1)
    if hasattr(env, "sensor_data_fresh"):
        values["sensor_data_fresh"] = env.sensor_data_fresh
    direction_y = torch.sign(env.sweep_dir[:, 1])
    values["object_forward_velocity_m_s"] = velocity_y * direction_y
    values["object_progress_m"] = env.sweep_dir[:, 1].abs() - (goal[:, 1] - target[:, 1]) * direction_y
    object_rotation = matrix_from_quat(objects.data.object_link_state_w[rows, ids, 3:7])
    values["object_tilt_deg"] = torch.rad2deg(torch.acos(object_rotation[:, 2, 2].clamp(-1.0, 1.0)))
    hand_rotation = matrix_from_quat(env.scene["ee_frame"].data.target_quat_w[:, 0])
    values["hand_up_error_deg"] = torch.rad2deg(torch.acos(hand_rotation[:, 2, 0].clamp(-1.0, 1.0)))
    if record_metrics is not None:
        record_metrics(values)
    if diagnostic_env is None:
        return result
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
