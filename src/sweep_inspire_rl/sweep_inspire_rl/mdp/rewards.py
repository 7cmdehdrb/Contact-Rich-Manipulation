"""Sweep-Policy rewards adapted to a single palmar pushing surface."""

from __future__ import annotations

import torch
from isaaclab.utils.math import matrix_from_quat

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


def pushing_target(env, command_name="target_goal_pos"):
    """Retain Sweep's goal kernel; gate progress by the upstream palm pose.

    The Robotiq wrist/finger proxy has no matching meaning on Inspire. The
    replacement gate requires an upstream hand and a palmar normal aligned to
    world +Y. Velocity shaping rewards rightward motion only.
    """
    objects = env.scene["object_collection"]
    target = objects.data.object_pos_w[:, 0]
    goal = env.command_manager.get_command(command_name)[:, :3]
    hand = palm_surface_position(env)
    distance = torch.linalg.vector_norm(goal - target, dim=-1)
    near_hand = torch.linalg.vector_norm(reaching_position(env) - hand, dim=-1) < 0.04
    # C is ahead of the physical pad by 37.3 mm. Testing C against the
    # object's center would turn this gate off before the pad can touch it.
    # A right-facing hand can still touch through a protruding thumb/carrier.
    # Require the actual palm pad, channel zero, for any positive sweep reward.
    # At the raised pre-push height and within this proximity gate, that pad
    # cannot be contacting the shelf's active board.
    palm_contact = palm_tactile_bits(env)[:, 0] > 0.5
    gate = near_hand & (hand[:, 1] <= upstream_surface_position(env) + 0.015) & (palm_alignment(env) > 0.9)
    gate &= palm_contact
    velocity_y = objects.data.object_lin_vel_w[:, 0, 1]
    velocity_reward = torch.where(
        velocity_y > 0.05,
        torch.where(velocity_y < 0.1, 0.5, -0.5),
        torch.where(velocity_y < -0.05, -0.5, 0.0),
    )
    return torch.where(
        distance < 0.03,
        gate.to(distance.dtype) * 2.0 * torch.exp(-5.0 * distance),
        gate.to(distance.dtype) * (1.0 - distance / 0.18 + velocity_reward),
    )


def hand_velocity_limit(env, threshold=1.0):
    """Resolve arm joints by name instead of the old first-six ordering."""
    term = env.action_manager.get_term("arm_action")
    return (env.scene["robot"].data.joint_vel[:, term._joint_ids].abs() > threshold).any(dim=1)
