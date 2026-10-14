"""Unit tests for single-stage entry, success, and failure kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_task_math():
    path = Path(__file__).parents[1] / "mdp" / "task_math.py"
    spec = importlib.util.spec_from_file_location("sweep_basic_task_math", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task_math = _load_task_math()


def test_success_requires_goal_and_low_speed_with_strict_thresholds():
    ready = task_math.target_reached_and_stopped_mask(
        torch.tensor([0.029, 0.030, 0.029]),
        torch.tensor([0.019, 0.019, 0.020]),
        distance_threshold=0.03,
        speed_threshold=0.02,
    )
    assert torch.equal(ready, torch.tensor([True, False, False]))


def test_entry_requires_continuous_contact_for_point_one_seconds():
    elapsed, entered = task_math.update_contact_entry_state(
        torch.tensor([0.08, 0.08, 0.00]),
        torch.tensor([False, False, True]),
        torch.tensor([True, False, False]),
        dt=0.02,
        dwell_time=0.10,
    )
    assert torch.allclose(elapsed, torch.tensor([0.10, 0.00, 0.00]))
    assert torch.equal(entered, torch.tensor([True, False, True]))


def test_midpoint_penalty_starts_at_half_and_stops_after_entry():
    penalty = task_math.midpoint_no_entry_mask(
        torch.tensor([374, 375, 500, 749]),
        torch.tensor([False, False, True, False]),
        max_episode_length=750,
        episode_fraction=0.50,
    )
    assert torch.equal(penalty, torch.tensor([False, True, False, True]))


def test_drop_and_speed_limits_use_strict_thresholds():
    dropped = task_math.object_drop_mask(
        torch.tensor([1.10, 1.14, 1.19]),
        torch.tensor([1.14, 1.14, 1.14]),
        maximum_height_change=0.04,
    )
    too_fast = task_math.speed_limit_mask(
        torch.tensor([[0.15, 0.0, 0.0], [0.151, 0.0, 0.0]]),
        maximum_speed=0.15,
    )
    assert torch.equal(dropped, torch.tensor([False, False, True]))
    assert torch.equal(too_fast, torch.tensor([False, True]))


def test_filtered_shelf_contact_force_uses_strict_threshold_and_fails_safe():
    forces = torch.zeros(4, 1, 2, 3)
    forces[1, 0, 0, 0] = 0.10
    forces[2, 0, 1, 1] = 0.101
    forces[3, 0, 0, 2] = float("nan")
    collision = task_math.contact_force_mask(forces, force_threshold=0.10)
    assert torch.equal(collision, torch.tensor([False, False, True, True]))

    target_contact = task_math.contact_force_mask(
        forces, force_threshold=0.10, fail_on_nonfinite=False
    )
    assert torch.equal(target_contact, torch.tensor([False, False, True, False]))


def test_speed_violation_must_persist_for_required_steps():
    count = torch.zeros(2, dtype=torch.long)
    count, done = task_math.update_sustained_violation(
        count, torch.tensor([True, True]), required_steps=3
    )
    assert torch.equal(done, torch.tensor([False, False]))
    count, done = task_math.update_sustained_violation(
        count, torch.tensor([True, False]), required_steps=3
    )
    assert torch.equal(count, torch.tensor([2, 0]))
    count, done = task_math.update_sustained_violation(
        count, torch.tensor([True, True]), required_steps=3
    )
    assert torch.equal(done, torch.tensor([True, False]))
