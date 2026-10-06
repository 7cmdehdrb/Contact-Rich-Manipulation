"""A physical contact reference overrides approach only, never the Cube goal."""

from dataclasses import fields, replace

import pytest
import torch

from test_push_terms import _fixture, _step, _touch


def _default_target(sample):
    target = sample.cube_pos_w-.03*sample.direction_w
    target[:, 2] += .03
    return target


def test_explicit_default_target_is_equivalent_to_baseline_without_override():
    baseline, sample = _fixture()
    tracker, override = _fixture()
    override = replace(override, approach_target_pos_w=_default_target(override))
    tracker.reset(None, override)
    for counter in range(1, 5):
        if counter == 2:
            _touch(sample)
            _touch(override)
        sample.cube_pos_w[:, 1] = override.cube_pos_w[:, 1] = .01*counter
        override.approach_target_pos_w[:] = _default_target(override)
        expected, actual = _step(baseline, sample, counter), _step(tracker, override, counter)
        for item in fields(expected):
            torch.testing.assert_close(getattr(actual, item.name), getattr(expected, item.name))


def test_contact_target_changes_approach_potential_without_changing_cube_goal_or_progress():
    baseline, sample = _fixture()
    tracker, override = _fixture()
    override = replace(override, approach_target_pos_w=_default_target(override))
    tracker.reset(None, override)
    fixed_goal = override.goal_pos_w.clone()
    override.approach_target_pos_w[:] = override.palm_pos_w
    closer = _step(tracker, override, 1)
    assert (closer.approach_delta > .9).all()
    assert not closer.progress_delta.any()
    torch.testing.assert_close(closer.cube_goal_distance_m, torch.full((2,), .25, dtype=torch.float64))
    torch.testing.assert_close(override.goal_pos_w, fixed_goal)
    _step(baseline, sample, 1)
    _touch(sample)
    _touch(override)
    sample.cube_pos_w[:, 1] = override.cube_pos_w[:, 1] = .05
    reference = _step(baseline, sample, 2)
    actual = _step(tracker, override, 2)
    torch.testing.assert_close(actual.progress_delta, reference.progress_delta)
    torch.testing.assert_close(actual.cube_goal_distance_m, reference.cube_goal_distance_m)
    assert not actual.success.any()


def test_optional_target_does_not_replace_cube_goal_success_geometry():
    tracker, sample = _fixture()
    sample = replace(sample, approach_target_pos_w=sample.palm_pos_w.clone())
    tracker.reset(None, sample)
    _touch(sample)
    # Reaching the selected contact reference is not Cube goal completion.
    for counter in range(1, 12):
        assert not _step(tracker, sample, counter).success.any()
    sample.cube_pos_w[:] = sample.goal_pos_w
    for counter in range(12, 22):
        final = _step(tracker, sample, counter)
    assert final.success.all()
    assert final.cube_goal_distance_m.sum() == 0


def test_subset_reset_keeps_other_target_record_and_does_not_repeat_old_approach_credit():
    tracker, sample = _fixture()
    sample = replace(sample, approach_target_pos_w=_default_target(sample))
    tracker.reset(None, sample)
    sample.approach_target_pos_w[0] = sample.palm_pos_w[0]
    first = _step(tracker, sample, 1)
    assert first.approach_delta[0] > .9 and first.approach_delta[1] == 0
    best = tracker._best_approach[0].clone()
    sample.approach_target_pos_w[1] = sample.palm_pos_w[1]+sample.palm_pos_w.new_tensor((0., .10, 0.))
    tracker.reset([1], sample)
    assert tracker._best_approach[0] == best
    assert first.approach_delta[0] > .9
    same_counter = _step(tracker, sample, 1)
    torch.testing.assert_close(same_counter.approach_delta[0], first.approach_delta[0])
    sample.approach_target_pos_w[1] = sample.palm_pos_w[1]
    next_state = _step(tracker, sample, 2)
    assert next_state.approach_delta[0] == 0 and next_state.approach_delta[1] > .8


def test_nonfinite_override_is_an_invalid_physical_state_and_cannot_pay_shaping():
    tracker, sample = _fixture()
    target = _default_target(sample)
    target[0, 2] = float('nan')
    _touch(sample)
    sample.cube_pos_w[:, 1] = .05
    state = _step(tracker, replace(sample, approach_target_pos_w=target), 1)
    assert state.failure.tolist() == state.invalid_state.tolist() == [True, False]
    assert state.additional_failure.tolist() == [False, False]
    assert state.progress_delta[0] == 0 and state.progress_delta[1] > 0
    for item in fields(state):
        assert torch.isfinite(getattr(state, item.name)).all(), item.name


@pytest.mark.parametrize('shape', ((2, 4), (3,), (1, 3)))
def test_approach_target_shape_is_validated(shape):
    tracker, sample = _fixture()
    with pytest.raises(ValueError, match='approach_target_pos_w'):
        _step(tracker, replace(sample, approach_target_pos_w=torch.zeros(shape)), 1)
