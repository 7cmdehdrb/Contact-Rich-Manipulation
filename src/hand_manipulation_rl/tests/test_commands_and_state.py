from __future__ import annotations

import math

import pytest
import torch

from hand_manipulation_rl.mdp.commands import (
    direction_from_angle,
    goal_from_command,
    is_allowed_sweep_angle,
    sample_sweep_angles,
    wrap_s_angle,
)
from hand_manipulation_rl.mdp.episode_state import (
    ACTION_DIM,
    DORSAL_MODE,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT,
    OBSERVATION_SLICES,
    PALM_MODE,
    EpisodeState,
)


def test_only_shelf_left_and_right_angles_are_allowed() -> None:
    degrees = torch.tensor([-90.0, 0.0, 90.0, 180.0, 270.0])
    allowed = is_allowed_sweep_angle(torch.deg2rad(degrees))
    expected = torch.tensor([True, False, True, False, True])
    assert torch.equal(allowed, expected)
    wrapped = wrap_s_angle(torch.tensor([3.0 * math.pi / 2.0, -math.pi / 2.0]))
    assert torch.allclose(wrapped, torch.tensor([-math.pi / 2.0, -math.pi / 2.0]))


def test_sampled_commands_are_equiprobable_exact_lateral_directions() -> None:
    generator = torch.Generator().manual_seed(1234)
    angles = sample_sweep_angles(20_000, dtype=torch.float64, generator=generator)
    assert torch.all(is_allowed_sweep_angle(angles))
    directions = direction_from_angle(angles)
    assert torch.all(directions[:, 0] == 0.0)
    assert torch.all(torch.abs(directions[:, 1]) == 1.0)
    assert torch.all(directions[:, 2] == 0.0)
    right_fraction = torch.mean((directions[:, 1] > 0.0).to(torch.float64))
    assert 0.48 < right_fraction < 0.52
    torch.testing.assert_close(
        torch.unique(angles),
        torch.tensor([-math.pi / 2.0, math.pi / 2.0], dtype=torch.float64),
    )


def test_goal_is_fixed_initial_position_plus_direction_times_distance() -> None:
    initial = torch.tensor([[0.1, 0.2, 0.3], [1.0, -2.0, 0.5]])
    angles = torch.tensor([-math.pi / 2.0, math.pi / 2.0])
    distances = torch.tensor([0.18, 0.18])
    goal = goal_from_command(initial, angles, distances)
    expected = initial + torch.tensor([[0.0, -0.18, 0.0], [0.0, 0.18, 0.0]])
    assert torch.allclose(goal, expected)


def test_observation_layout_is_contiguous_and_exactly_57d() -> None:
    assert OBSERVATION_DIM == 57
    assert ACTION_DIM == 8
    offset = 0
    for name, width in OBSERVATION_LAYOUT:
        assert OBSERVATION_SLICES[name] == slice(offset, offset + width)
        offset += width
    assert offset == OBSERVATION_DIM


def test_episode_spec_is_single_source_and_subset_reset_preserves_command() -> None:
    state = EpisodeState.allocate(3, dtype=torch.float64)
    state.set_episode_spec(
        None,
        mode=torch.tensor([PALM_MODE, DORSAL_MODE, PALM_MODE]),
        angle_s=torch.tensor([-math.pi / 2.0, math.pi / 2.0, -math.pi / 2.0], dtype=torch.float64),
        distance=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        initial_object_position_s=torch.tensor(
            [[0.0, 0.0, 0.2], [0.1, 0.2, 0.2], [-0.1, 0.1, 0.2]], dtype=torch.float64
        ),
    )
    saved_goal = state.goal_position_s.clone()
    state.record_action(torch.ones(3, 8, dtype=torch.float64))
    state.sensor_valid[:] = True
    state.latch_failures(
        topple=torch.tensor([True, False, False]),
        board_contact=torch.tensor([False, True, False]),
        height=torch.tensor([False, False, True]),
    )
    state.update_board_force_peak(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    state.reset_transient(torch.tensor([1]))

    assert torch.equal(state.previous_action_valid, torch.tensor([True, False, True]))
    assert torch.equal(state.sensor_valid, torch.tensor([True, False, True]))
    assert torch.equal(state.board_contact_latched, torch.tensor([False, False, False]))
    assert torch.equal(state.topple_latched, torch.tensor([True, False, False]))
    assert torch.equal(state.height_latched, torch.tensor([False, False, True]))
    assert state.board_force_peak.tolist() == [1.0, 0.0, 3.0]
    assert torch.equal(state.goal_position_s, saved_goal)


def test_episode_spec_rejects_non_lateral_direction_and_invalid_mode() -> None:
    state = EpisodeState.allocate(1)
    with pytest.raises(ValueError, match="shelf-left or shelf-right"):
        state.set_episode_spec(
            None,
            mode=PALM_MODE,
            angle_s=0.0,
            distance=0.1,
            initial_object_position_s=torch.zeros(1, 3),
        )
    with pytest.raises(ValueError, match="mode"):
        state.set_episode_spec(
            None,
            mode=3,
            angle_s=math.pi / 2.0,
            distance=0.1,
            initial_object_position_s=torch.zeros(1, 3),
        )
