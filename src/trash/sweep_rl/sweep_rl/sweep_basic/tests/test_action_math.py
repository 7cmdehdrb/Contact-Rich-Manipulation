"""Unit tests for simulator-independent arm action conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_action_math():
    path = Path(__file__).parents[1] / "mdp" / "action_math.py"
    spec = importlib.util.spec_from_file_location("sweep_basic_action_math", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


action_math = _load_action_math()


def test_incremental_target_is_current_relative_and_normalized():
    current = torch.tensor([[0.20, -1.00, 1.50]])
    action = torch.tensor([[1.00, -0.50, 2.00]])
    limits = torch.tensor([[[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]])
    normalized, target = action_math.incremental_joint_target(
        current, action, limits, scale=0.05
    )
    assert torch.allclose(normalized, torch.tensor([[1.0, -0.5, 1.0]]))
    assert torch.allclose(target, torch.tensor([[0.25, -1.025, 1.55]]))


def test_incremental_target_respects_joint_limits_and_sanitizes_nonfinite():
    current = torch.tensor([[0.99, -0.99, 0.25]])
    action = torch.tensor([[1.0, -1.0, float("nan")]])
    limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]])
    normalized, target = action_math.incremental_joint_target(
        current, action, limits, scale=0.05
    )
    assert torch.allclose(normalized, torch.tensor([[1.0, -1.0, 0.0]]))
    assert torch.allclose(target, torch.tensor([[1.0, -1.0, 0.25]]))
