from __future__ import annotations

import torch

from hand_manipulation_rl.mdp.terminations import (
    TerminationReason,
    latch_threshold,
    resolve_terminations,
    success_from_positions,
    threshold_exceeded,
)


def test_success_uses_strict_3d_one_centimeter_threshold() -> None:
    object_position = torch.tensor([[0.009, 0.0, 0.0], [0.01, 0.0, 0.0]])
    goal_position = torch.zeros_like(object_position)
    assert torch.equal(success_from_positions(object_position, goal_position), torch.tensor([True, False]))


def test_hard_threshold_is_strict_and_latched() -> None:
    values = torch.tensor([0.9, 1.0, 1.1])
    assert torch.equal(threshold_exceeded(values, 1.0), torch.tensor([False, False, True]))
    previous = torch.tensor([True, False, False])
    assert torch.equal(latch_threshold(previous, values, 1.0), torch.tensor([True, False, True]))


def test_failure_wins_over_success_and_timeout_is_truncation() -> None:
    decision = resolve_terminations(
        success_candidate=torch.tensor([True, True, False, False, True]),
        topple=torch.tensor([True, False, False, False, False]),
        board_contact=torch.tensor([False, False, False, False, False]),
        height_limit=torch.tensor([False, False, False, False, False]),
        timeout=torch.tensor([True, True, True, False, False]),
        invalid_simulation=torch.tensor([False, False, False, True, True]),
    )
    assert torch.equal(decision.terminated, torch.tensor([True, True, False, True, True]))
    assert torch.equal(decision.truncated, torch.tensor([False, False, True, False, False]))
    assert torch.equal(decision.succeeded, torch.tensor([False, True, False, False, False]))
    assert torch.equal(decision.failed, torch.tensor([True, False, False, False, False]))
    assert torch.equal(decision.errored, torch.tensor([False, False, False, True, True]))
    assert decision.reason.tolist() == [
        int(TerminationReason.TOPPLE),
        int(TerminationReason.SUCCESS),
        int(TerminationReason.TIMEOUT),
        int(TerminationReason.INVALID_SIMULATION),
        int(TerminationReason.INVALID_SIMULATION),
    ]


def test_primary_failure_reason_is_deterministic() -> None:
    decision = resolve_terminations(
        success_candidate=torch.tensor([True]),
        topple=torch.tensor([True]),
        board_contact=torch.tensor([True]),
        height_limit=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    assert decision.reason.item() == int(TerminationReason.TOPPLE)
    assert decision.failed.item()
    assert not decision.succeeded.item()
    assert not decision.timed_out.item()

