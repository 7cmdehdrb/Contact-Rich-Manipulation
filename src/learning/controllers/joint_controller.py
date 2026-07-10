"""Joint-space action conversion."""

from __future__ import annotations

import torch


class JointController:
    """Converts policy actions to bounded joint-position deltas."""

    def __init__(self, action_scale: float, max_delta: float) -> None:
        self.action_scale = action_scale
        self.max_delta = max_delta

    def to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != 6:
            raise ValueError(f"Joint actions must have 6 dimensions, got {actions.shape[-1]}.")
        return torch.clamp(actions, -1.0, 1.0) * self.action_scale
