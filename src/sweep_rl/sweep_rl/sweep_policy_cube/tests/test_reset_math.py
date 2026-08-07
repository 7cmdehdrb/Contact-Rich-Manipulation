"""Tests for the original-slot Cube reset sampler."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "mdp" / "reset_math.py"
SPEC = importlib.util.spec_from_file_location("sweep_policy_cube_reset_math", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
reset_math = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reset_math)


def test_each_environment_uses_its_selected_slot_and_jitter():
    slots = torch.tensor(
        [
            [-0.70, -0.20, 1.11],
            [-0.70, 0.00, 1.11],
            [-0.70, 0.20, 1.11],
            [-0.55, -0.20, 1.11],
            [-0.55, 0.00, 1.11],
            [-0.55, 0.20, 1.11],
        ]
    )
    indices = torch.tensor([0, 2, 5])
    jitter = torch.tensor([[0.02, -0.02], [0.0, 0.01], [-0.01, 0.0]])
    result = reset_math.sample_slot_positions(indices, slots, jitter, 1.11)
    expected = torch.tensor(
        [[-0.68, -0.22, 1.11], [-0.70, 0.21, 1.11], [-0.56, 0.20, 1.11]]
    )
    torch.testing.assert_close(result, expected)


def test_center_height_overrides_slot_z_without_mutating_slots():
    slots = torch.tensor([[-0.55, 0.00, 99.0]])
    original = slots.clone()
    result = reset_math.sample_slot_positions(
        torch.tensor([0]), slots, torch.zeros(1, 2), 1.11
    )
    torch.testing.assert_close(result[0, 2], torch.tensor(1.11))
    torch.testing.assert_close(slots, original)
