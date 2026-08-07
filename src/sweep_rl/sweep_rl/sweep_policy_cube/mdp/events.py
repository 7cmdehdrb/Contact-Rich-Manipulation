"""Reset events for the single Cube."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import random_yaw_orientation

from .reset_math import sample_slot_positions


def reset_target_from_slots(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    slots: tuple[tuple[float, float, float], ...] = (),
    position_jitter: float = 0.02,
    center_height: float = 1.08,
) -> None:
    """Place each Cube in one of the original six Sweep-Policy slots."""
    target: RigidObject = env.scene[asset_cfg.name]
    count = env_ids.numel()
    slot_tensor = torch.tensor(slots, dtype=torch.float32, device=env.device)
    slot_indices = torch.randint(
        low=0, high=slot_tensor.shape[0], size=(count,), device=env.device
    )
    jitter = torch.empty(count, 2, device=env.device).uniform_(
        -position_jitter, position_jitter
    )
    positions = sample_slot_positions(slot_indices, slot_tensor, jitter, center_height)
    positions += env.scene.env_origins[env_ids]
    orientations = random_yaw_orientation(count, device=env.device)
    target.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=env_ids
    )
    target.write_root_velocity_to_sim(
        torch.zeros(count, 6, dtype=torch.float32, device=env.device),
        env_ids=env_ids,
    )
