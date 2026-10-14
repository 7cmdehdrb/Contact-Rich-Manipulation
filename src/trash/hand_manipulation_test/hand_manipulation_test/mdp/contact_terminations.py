"""Table contact is terminal and takes precedence over the inherited timeout."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp import time_out

from ..contact_sensors import table_contact_mask


def table_contact_failure(env) -> torch.Tensor:
    return table_contact_mask(env)


def contact_time_out(env) -> torch.Tensor:
    return time_out(env) & ~table_contact_failure(env)


__all__ = ["table_contact_failure", "contact_time_out"]
