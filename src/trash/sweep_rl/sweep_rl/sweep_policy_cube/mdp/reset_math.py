"""Pure tensor helpers for Cube reset sampling."""

from __future__ import annotations

import torch


def sample_slot_positions(
    slot_indices: torch.Tensor,
    slots: torch.Tensor,
    jitter: torch.Tensor,
    center_height: float,
) -> torch.Tensor:
    positions = slots[slot_indices].clone()
    positions[:, :2] += jitter
    positions[:, 2] = center_height
    return positions
