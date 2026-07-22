"""Reset-time selection and arrangement of shelf objects."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObjectCollection
from isaaclab.utils.math import random_yaw_orientation


def randomize_shelf_scene(
    env,
    env_ids: torch.Tensor,
    command_name: str,
    ceiling_height: float,
) -> None:
    command = env.command_manager.get_term(command_name)
    objects: RigidObjectCollection = env.scene[command.cfg.asset_name]
    num_objects = len(command.cfg.object_widths)
    count = len(env_ids)
    if count == 0:
        return

    target_ids = torch.randint(0, num_objects, (count,), device=env.device)
    sweep_sign = 2.0 * torch.randint(0, 2, (count,), device=env.device).float() - 1.0
    command.target_ids[env_ids] = target_ids
    widths = torch.tensor(command.cfg.object_widths, device=env.device)
    command.target_width[env_ids, 0] = widths[target_ids]
    command.sweep_direction_w[env_ids] = 0.0
    command.sweep_direction_w[env_ids, 1] = command.cfg.sweep_distance * sweep_sign

    state = objects.data.default_object_state[env_ids].clone()
    state[:, :, :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    state[:, :, :2] += torch.empty(
        count, num_objects, 2, device=env.device
    ).uniform_(-0.02, 0.02)
    state[:, :, 3:7] = random_yaw_orientation(
        count * num_objects, device=env.device
    ).view(count, num_objects, 4)
    state[:, :, 7:] = 0.0

    object_ids = torch.arange(num_objects, device=env.device).unsqueeze(0)
    object_rows = object_ids // 3
    object_cols = object_ids % 3
    target_rows = (target_ids // 3).unsqueeze(1)
    target_cols = (target_ids % 3).unsqueeze(1)
    side_cols = target_cols + sweep_sign.long().unsqueeze(1)
    in_front = (object_rows < target_rows) & (object_cols == target_cols)
    on_sweep_side = (object_rows == target_rows) & (object_cols == side_cols)
    front_diagonal = (object_rows < target_rows) & (object_cols == side_cols)
    hide = (in_front | on_sweep_side | front_diagonal) & (
        object_ids != target_ids.unsqueeze(1)
    )
    hidden_z = env.scene.env_origins[env_ids, 2].unsqueeze(1) + ceiling_height
    state[:, :, 2] = torch.where(hide, hidden_z, state[:, :, 2])

    objects.write_object_state_to_sim(state, env_ids=env_ids)
