"""Small continuous reward for real Cube contact on physical palmar pads."""

from __future__ import annotations

import torch

from ..contact_sensors import cube_palm_contact_mask, table_contact_mask


def palm_cube_contact_reward(env) -> torch.Tensor:
    """Return one while palm-Cube contact persists, zero after table failure.

    Reset rows cannot reuse contact data from their previous episode. The
    thresholded signal comes exclusively from Cube-filtered physical pad
    forces; broad hand-link contact cannot impersonate a tactile pad.
    """

    length = getattr(env, "episode_length_buf", None)
    if length is None:
        return torch.zeros(env.num_envs, device=env.device)
    live = length > 0
    contact = cube_palm_contact_mask(env)
    failed = table_contact_mask(env)
    return (contact & live & ~failed).to(dtype=torch.float32)


__all__ = ["palm_cube_contact_reward"]
