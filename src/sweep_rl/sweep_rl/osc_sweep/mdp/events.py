"""Reset randomization for the sweep task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def reset_target_object(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Randomize cube XY position/yaw and clear its velocities."""
    target: RigidObject = env.scene[asset_cfg.name]
    count = len(env_ids)
    if count == 0:
        return

    default_state = target.data.default_root_state[env_ids].clone()
    positions = default_state[:, :3]
    positions += env.scene.env_origins[env_ids]
    positions[:, 0] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("x", (0.0, 0.0))
    )
    positions[:, 1] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("y", (0.0, 0.0))
    )
    positions[:, 2] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("z", (0.0, 0.0))
    )

    yaw = torch.empty(count, device=env.device).uniform_(
        *pose_range.get("yaw", (0.0, 0.0))
    )
    zero = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zero, zero, yaw)

    velocities = default_state[:, 7:13]
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        low, high = velocity_range.get(key, (0.0, 0.0))
        velocities[:, index] += torch.empty(
            count, device=env.device
        ).uniform_(low, high)

    target.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=env_ids
    )
    target.write_root_velocity_to_sim(velocities, env_ids=env_ids)
