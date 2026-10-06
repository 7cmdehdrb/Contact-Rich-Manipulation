"""Additional physical failure shares the existing cached Push transition."""

from dataclasses import fields, replace

import pytest
import torch

from test_push_terms import _environment, _fixture, _step, _touch, rewards, terminations


def test_absent_additional_failure_preserves_every_base_transition_field():
    original, sample = _fixture()
    explicit, other = _fixture()
    other = replace(other, additional_failure=torch.zeros(2, dtype=torch.bool))
    for counter in range(1, 5):
        if counter == 2:
            _touch(sample)
            _touch(other)
        sample.cube_pos_w[:, 1] = other.cube_pos_w[:, 1] = counter*.025
        expected, actual = _step(original, sample, counter), _step(explicit, other, counter)
        for item in fields(expected):
            torch.testing.assert_close(getattr(actual, item.name), getattr(expected, item.name))
    assert original._failure_flags.shape == explicit._failure_flags.shape == (2, 4)


def test_additional_failure_suppresses_positive_shaping_and_keeps_original_causes_distinct():
    tracker, sample = _fixture(3)
    _touch(sample)
    sample.cube_pos_w[:, 1] = .05
    sample = replace(sample, additional_failure=torch.tensor([True, False, True]),
                     live=torch.tensor([True, True, False]))
    state = _step(tracker, sample, 1)
    assert state.failure.tolist() == [True, False, False]
    assert state.additional_failure.tolist() == [True, False, False]
    assert state.first_contact.tolist() == [False, True, False]
    assert state.maintaining_contact.tolist() == [0., 1., 0.]
    assert state.progress_delta[0] == 0 and state.progress_delta[1] > 0
    assert not state.approach_delta.any() and not state.success.any()
    assert not state.table_failure.any() and not state.fall_failure.any()
    assert not state.footprint_failure.any() and not state.invalid_state.any()
    assert state.palm_cube_contact.tolist() == [True, True, False]
    assert not tracker._failure_flags.any()


def test_additional_failure_precedes_settled_success_and_timeout_in_real_terms():
    tracker, sample = _fixture(3)
    sample.cube_pos_w[:2] = sample.goal_pos_w[:2]
    _touch(sample)
    for counter in range(1, 10):
        assert not _step(tracker, sample, counter).success.any()
    sample = replace(sample, additional_failure=torch.tensor([True, False, False]))
    env = _environment(tracker, sample, counter=10)
    env.episode_length_buf[:] = 500
    assert terminations.push_failure(env).tolist() == [True, False, False]
    assert terminations.push_success(env).tolist() == [False, True, False]
    assert terminations.push_time_out(env).tolist() == [False, False, True]
    for func in (rewards.push_approach_reward, rewards.push_progress_reward,
                 rewards.push_first_contact_reward, rewards.push_contact_reward,
                 rewards.push_success_reward):
        assert func(env)[0] == 0, func.__name__
    assert rewards.push_failure_reward(env)[0] == -50
    assert tracker._cached.settled_time_s[0] == 0
    assert tracker._cached.settled_time_s[1] == pytest.approx(.2)


def test_additional_failure_latches_until_selected_reset_and_preserves_returned_state():
    tracker, sample = _fixture(3)
    sample = replace(sample, additional_failure=torch.tensor([True, True, False]))
    failed = _step(tracker, sample, 1)
    recovered = replace(sample, additional_failure=torch.zeros(3, dtype=torch.bool))
    latched = _step(tracker, recovered, 2)
    assert latched.failure.tolist() == latched.additional_failure.tolist() == [True, True, False]
    tracker.reset([0], recovered)
    assert tracker._additional_failure.tolist() == [False, True, False]
    assert failed.additional_failure.tolist() == [True, True, False]
    # Recompute within the same global physics counter after partial reset:
    # reset row clears its failure, untouched rows retain the old transition.
    reset_sample = replace(recovered, live=torch.tensor([False, True, True]))
    reset_state = _step(tracker, reset_sample, 2)
    assert reset_state.failure.tolist() == reset_state.additional_failure.tolist() == [False, True, False]
    for item in fields(latched):
        torch.testing.assert_close(getattr(reset_state, item.name)[1:], getattr(latched, item.name)[1:])
    tracker.reset([1], reset_sample)
    assert not _step(tracker, replace(recovered, live=torch.ones(3, dtype=torch.bool)), 3).failure.any()


def test_repeated_counter_never_advances_an_additional_failure_or_settling_timer():
    tracker, sample = _fixture()
    _touch(sample)
    sample.cube_pos_w[:] = sample.goal_pos_w
    sample = replace(sample, additional_failure=torch.zeros(2, dtype=torch.bool))
    first = _step(tracker, sample, 1)
    sample.additional_failure[0] = True
    for _ in range(5):
        assert _step(tracker, sample, 1) is first
    assert not first.additional_failure.any()
    torch.testing.assert_close(first.settled_time_s, torch.full((2,), .02, dtype=torch.float64))
    next_state = _step(tracker, sample, 2)
    assert next_state.additional_failure.tolist() == [True, False]
    torch.testing.assert_close(next_state.settled_time_s, torch.tensor([0., .04], dtype=torch.float64))


@pytest.mark.parametrize("mask", (torch.zeros(2, 1, dtype=torch.bool), torch.tensor(True),
                                   torch.tensor([float('nan'), 0.]), torch.tensor([0, 1])))
def test_additional_failure_rejects_wrong_shape_or_nonboolean_values(mask):
    tracker, sample = _fixture()
    with pytest.raises(ValueError, match="additional_failure"):
        _step(tracker, replace(sample, additional_failure=mask), 1)
