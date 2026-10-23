"""Single-object resets with independently selectable left/right sweep events."""

from __future__ import annotations

import math as math
import torch as torch

def randomize_single_object_scene(
    env,
    env_ids,
    sweep_direction="both",
    object_radius=0.04,
    position_noise=0.02,
    initial_position=(-0.75, 0.0, 1.05),
):
    """Spawn one cylinder at the source single-object position plus XY noise.

    Z remains the shelf surface height. Direction is sampled independently per
    environment: ``left`` means world -Y, ``right`` means world +Y.
    """
    if sweep_direction not in ("both", "left", "right"):
        raise ValueError("sweep_direction must be 'both', 'left', or 'right'")
    if object_radius <= 0.0 or position_noise < 0.0:
        raise ValueError("object_radius must be positive and position_noise nonnegative")
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    count = env_ids.numel()
    if count == 0:
        return
    objects = env.scene["object_collection"]
    if objects.num_objects != 1:
        raise RuntimeError("The baseline requires exactly one cylinder in object_collection")
    positions = torch.tensor(initial_position, device=env.device).expand(count, 3).clone()
    positions[:, :2] += torch.empty((count, 2), device=env.device).uniform_(
        -position_noise, position_noise
    )
    positions += env.scene.env_origins[env_ids]
    yaw = torch.empty(count, device=env.device).uniform_(-math.pi, math.pi)
    rotations = torch.zeros((count, 4), device=env.device)
    rotations[:, 0] = torch.cos(yaw / 2.0)
    rotations[:, 3] = torch.sin(yaw / 2.0)
    states = torch.cat((positions, rotations, torch.zeros((count, 6), device=env.device)), dim=-1)
    objects.write_object_link_state_to_sim(
        states.unsqueeze(1), env_ids=env_ids,
        object_ids=torch.tensor([0], device=env.device, dtype=torch.long),
    )
    env.target_id[env_ids, 0] = 0
    env.target_width[env_ids, 0] = object_radius
    if sweep_direction == "both":
        direction = 2.0 * torch.randint(0, 2, (count,), device=env.device) - 1.0
    else:
        direction = torch.full((count,), -1.0 if sweep_direction == "left" else 1.0, device=env.device)
    env.sweep_dir[env_ids] = 0.0
    env.sweep_dir[env_ids, 1] = 0.18 * direction


def spawn_left_sweep(env, env_ids, **kwargs):
    """Convenience event for a left-only experiment."""
    randomize_single_object_scene(env, env_ids, sweep_direction="left", **kwargs)


def spawn_right_sweep(env, env_ids, **kwargs):
    """Convenience event for a right-only experiment."""
    randomize_single_object_scene(env, env_ids, sweep_direction="right", **kwargs)
