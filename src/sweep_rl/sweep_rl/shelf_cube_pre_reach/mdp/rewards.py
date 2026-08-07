"""Shelf-safety reward terms for Cube pre-reaching."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def shelf_collision(
    env,
    threshold: float = 0.005,
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
) -> torch.Tensor:
    """Flag shelf motion caused by a robot collision.

    This follows Sweep-Policy's binary shelf-motion term.  The shelf is
    considered hit when its displacement from the configured default position
    plus its 6-D root velocity norm exceeds ``threshold``.
    """

    shelf: RigidObject = env.scene[shelf_cfg.name]
    initial_pos_w = (
        shelf.data.default_root_state[:, :3] + env.scene.env_origins
    )
    position_delta = shelf.data.root_pos_w - initial_pos_w
    motion = torch.linalg.norm(position_delta, dim=-1) + torch.linalg.norm(
        shelf.data.root_vel_w, dim=-1
    )
    return (motion > threshold).to(dtype=torch.float32)
