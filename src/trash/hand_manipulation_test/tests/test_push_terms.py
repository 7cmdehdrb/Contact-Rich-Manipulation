"""Physical pushing-history and manager-order regressions without Isaac Sim."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"
TEST_PACKAGE = "_hand_manipulation_push_terms_fixture"
package = ModuleType(TEST_PACKAGE)
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules[TEST_PACKAGE] = package
mdp = ModuleType(f"{TEST_PACKAGE}.mdp")
mdp.__path__ = [str(PACKAGE_ROOT / "mdp")]
sys.modules[mdp.__name__] = mdp


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


math_tools = _load(f"{TEST_PACKAGE}.action_math", "action_math.py")
state_tools = _load(f"{TEST_PACKAGE}.push_state", "push_state.py")
rewards = _load(f"{TEST_PACKAGE}.mdp.push_rewards", "mdp/push_rewards.py")
terminations = _load(f"{TEST_PACKAGE}.mdp.push_terminations", "mdp/push_terminations.py")


def _fixture(count=2, *, sigma_fraction=.5):
    identity = torch.tensor((1., 0., 0., 0.), dtype=torch.float64).expand(count, -1).clone()
    cube = torch.tensor((0., 0., .05), dtype=torch.float64).expand(count, -1).clone()
    palm = torch.tensor((0., -.17, .08), dtype=torch.float64).expand(count, -1).clone()
    direction = torch.tensor((0., 1., 0.), dtype=torch.float64).expand(count, -1).clone()
    table_forces = torch.zeros(count, 2, 1, 1, 3, dtype=torch.float64)
    table_forces[..., 2] = 9.81
    sample = state_tools.PushSnapshot(
        initial_cube_pos_w=cube.clone(), goal_pos_w=cube + .25 * direction,
        direction_w=direction, distance_m=torch.full((count,), .25, dtype=torch.float64),
        cube_pos_w=cube, cube_quat_w=identity.clone(), cube_lin_vel_w=torch.zeros_like(cube),
        palm_pos_w=palm, hand_quat_w=identity.clone(),
        cube_palm_forces_w_history=torch.zeros(count, 2, 1, 17, 3, dtype=torch.float64),
        cube_table_forces_w_history=table_forces, robot_table_failure=torch.zeros(count, dtype=torch.bool),
        table_pos_w=torch.zeros_like(cube), table_quat_w=identity.clone(), live=torch.ones(count, dtype=torch.bool),
    )
    tracker = state_tools.PushStateTracker(
        state_tools.PushStateConfig(progress_sigma_fraction=sigma_fraction), count, "cpu", dtype=torch.float64
    )
    tracker.reset(None, sample)
    return tracker, sample


def _touch(sample, rows=slice(None), history=slice(None), pad=0, force=(0., 1., 0.)):
    sample.cube_palm_forces_w_history[rows, history, 0, pad] = torch.tensor(force, dtype=torch.float64)


def _step(tracker, sample, counter):
    return tracker.update(sample, physics_counter=counter, step_dt=.02)


def _environment(tracker, sample, counter=1):
    env = SimpleNamespace(step_dt=.02, _sim_step_counter=counter,
                          episode_length_buf=torch.ones(len(sample.distance_m), dtype=torch.long), max_episode_length=500)
    command = SimpleNamespace(state=lambda: tracker.update(sample, physics_counter=env._sim_step_counter, step_dt=env.step_dt))
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return env


def test_contact_requires_inclusive_individual_pair_and_support_in_same_substep():
    tracker, sample = _fixture(4)
    sample.cube_table_forces_w_history[:] = 0
    sample.cube_table_forces_w_history[:, 0, 0, 0, 2] = .01
    _touch(sample, history=1, force=(0., .01, 0.))
    sample.cube_table_forces_w_history[1:, 1, 0, 0, 2] = .01
    # Opposing pads never cancel a qualifying forward pad.
    _touch(sample, rows=2, history=1, pad=1, force=(0., -.01, 0.))
    # Even seventeen almost-threshold forward forces cannot create contact.
    sample.cube_palm_forces_w_history[3, :, 0, :, 1] = .009
    state = _step(tracker, sample, 1)
    assert state.grounded.tolist() == [True] * 4
    assert state.valid_push_contact.tolist() == [False, True, True, False]
    assert state.first_contact.tolist() == [False, True, True, False]


def test_plane_and_individual_force_use_full_3d_angle_without_force_cancellation():
    tracker, sample = _fixture(4)
    _touch(sample)
    sample.hand_quat_w[0] = math_tools.quaternion_from_rotation_vector(torch.tensor((math.pi / 2, 0., 0.), dtype=torch.float64))
    _touch(sample, rows=1, force=(0., 0., 2.))
    # A dominant vertical pad cannot hide a different qualifying forward pad.
    _touch(sample, rows=2, pad=1, force=(0., 0., 100.))
    sample.hand_quat_w[3] = math_tools.quaternion_from_rotation_vector(torch.tensor((0., 0., math.pi / 6), dtype=torch.float64))
    state = _step(tracker, sample, 1)
    assert state.valid_push_contact.tolist() == [False, False, True, True]
    assert state.palm_alignment_cos[0] == pytest.approx(0., abs=1.e-12)
    assert state.alignment_penalty[0] == pytest.approx(-.5)


def test_alignment_penalty_applies_before_contact_and_has_no_stationary_positive_alignment():
    tracker, sample = _fixture(2)
    sample.hand_quat_w[0] = math_tools.quaternion_from_rotation_vector(torch.tensor((math.pi / 2, 0., 0.), dtype=torch.float64))
    env = _environment(tracker, sample)
    torch.testing.assert_close(rewards.push_alignment_reward(env), torch.tensor((-.5, 0.), dtype=torch.float64), atol=1.e-12, rtol=0)
    torch.testing.assert_close(rewards.push_contact_reward(env), torch.zeros(2, dtype=torch.float64))


def test_first_valid_contact_suppresses_approach_and_is_one_shot_but_maintenance_persists():
    tracker, sample = _fixture()
    sample.palm_pos_w[:, 1] = -.03
    _touch(sample)
    first = _step(tracker, sample, 1)
    assert first.first_contact.all()
    assert not first.approach_delta.any()
    assert first.maintaining_contact.tolist() == [1., 1.]
    repeated = _step(tracker, sample, 2)
    assert not repeated.first_contact.any()
    assert repeated.maintaining_contact.tolist() == [1., 1.]
    sample.cube_palm_forces_w_history[:] = 0
    assert not _step(tracker, sample, 3).maintaining_contact.any()
    _touch(sample)
    retouch = _step(tracker, sample, 4)
    assert not retouch.first_contact.any()
    assert retouch.maintaining_contact.all()
    assert not retouch.approach_delta.any()


def test_approach_uses_current_cube_upstream_face_and_reward_cannot_repeat_old_record():
    tracker, sample = _fixture()
    sample.palm_pos_w[:, 1] = -.06
    forward = _step(tracker, sample, 1)
    assert (forward.approach_delta > 0).all()
    sample.palm_pos_w[:, 1] = -.17
    assert not _step(tracker, sample, 2).approach_delta.any()
    sample.palm_pos_w[:, 1] = -.06
    assert not _step(tracker, sample, 3).approach_delta.any()
    # Move Cube without contact and preserve the same relative gap: approach
    # tracks its actual face and never the fixed initial Cube/EEF point.
    sample.cube_pos_w[:, 1] = .02
    sample.palm_pos_w[:, 1] = -.04
    assert not _step(tracker, sample, 4).approach_delta.any()


def test_progress_only_new_grounded_contact_records_and_no_recovery_loop_credit():
    tracker, sample = _fixture()
    sample.cube_pos_w[:, 1] = .05
    assert not _step(tracker, sample, 1).progress_delta.any()
    _touch(sample)
    assert not _step(tracker, sample, 2).progress_delta.any()  # no retroactive drift reward
    sample.cube_pos_w[:, 1] = .10
    torch.testing.assert_close(_step(tracker, sample, 3).progress_delta,
                               torch.full((2,), .25558008512898667, dtype=torch.float64))
    sample.cube_palm_forces_w_history[:] = 0
    sample.cube_pos_w[:, 1] = .06
    torch.testing.assert_close(_step(tracker, sample, 4).backslide_delta, torch.full((2,), -.16, dtype=torch.float64))
    _touch(sample)
    sample.cube_pos_w[:, 1] = .10
    assert not _step(tracker, sample, 5).progress_delta.any()
    sample.cube_pos_w[:, 1] = .11
    torch.testing.assert_close(_step(tracker, sample, 6).progress_delta,
                               torch.full((2,), .0399531191291731, dtype=torch.float64))


def test_airborne_and_coasting_records_are_not_later_payable():
    tracker, sample = _fixture()
    _touch(sample)
    _step(tracker, sample, 1)
    sample.cube_table_forces_w_history[:] = 0
    sample.cube_pos_w[:, 1] = .05
    airborne = _step(tracker, sample, 2)
    assert not airborne.progress_delta.any() and not airborne.maintaining_contact.any()
    sample.cube_table_forces_w_history[..., 2] = 9.81
    assert not _step(tracker, sample, 3).progress_delta.any()
    sample.cube_palm_forces_w_history[:] = 0
    sample.cube_pos_w[:, 1] = .10
    assert not _step(tracker, sample, 4).progress_delta.any()
    _touch(sample)
    assert not _step(tracker, sample, 5).progress_delta.any()


def test_cache_supports_termination_before_rewards_without_duplicate_first_contact_or_timer():
    tracker, sample = _fixture()
    sample.cube_pos_w[:] = sample.goal_pos_w
    _touch(sample)
    env = _environment(tracker, sample)
    assert not terminations.push_success(env).any()
    first = _step(tracker, sample, 1)
    assert first.settled_time_s.tolist() == [.02, .02]
    assert _step(tracker, sample, 1) is first
    for _ in range(4):
        assert not terminations.push_success(env).any()
        torch.testing.assert_close(rewards.push_first_contact_reward(env), torch.full((2,), 50.))
    assert _step(tracker, sample, 1).settled_time_s.tolist() == [.02, .02]
    for counter in range(2, 11):
        env._sim_step_counter = counter
        state = _step(tracker, sample, counter)
    assert state.success.all()
    assert not state.first_contact.any()
    assert state.push_seen.all()


def test_goal_requires_prior_side_contact_grounding_3d_distance_speed_and_continuous_hold():
    tracker, sample = _fixture()
    sample.cube_pos_w[:] = sample.goal_pos_w
    for counter in range(1, 12):
        assert not _step(tracker, sample, counter).success.any()
    _touch(sample)
    assert _step(tracker, sample, 12).contact_seen.all()
    sample.cube_pos_w[:, 2] += .012
    for counter in range(13, 25):
        assert not _step(tracker, sample, counter).success.any()
    sample.cube_pos_w[:, 2] -= .012
    sample.cube_lin_vel_w[:, 1] = .02001
    assert not _step(tracker, sample, 25).settled_time_s.any()
    sample.cube_lin_vel_w[:, 1] = .02
    for counter in range(26, 35):
        assert not _step(tracker, sample, counter).success.any()
    sample.cube_table_forces_w_history[:] = 0
    assert not _step(tracker, sample, 35).settled_time_s.any()
    sample.cube_table_forces_w_history[..., 2] = .01
    for counter in range(36, 46):
        final = _step(tracker, sample, counter)
    assert final.success.all()


def test_actual_rotated_cube_footprint_uses_table_pose_and_margin():
    tracker, sample = _fixture(2)
    table_rotation = math_tools.quaternion_from_rotation_vector(torch.tensor((0., 0., math.pi / 2), dtype=torch.float64))
    sample.table_quat_w[:] = table_rotation
    sample.table_pos_w[:] = torch.tensor((2., 3., .4), dtype=torch.float64)
    local_cube = torch.tensor((.145, 0., .05), dtype=torch.float64).expand(2, -1)
    sample.cube_pos_w[:] = sample.table_pos_w + math_tools.quaternion_rotate(sample.table_quat_w, local_cube)
    sample.initial_cube_pos_w[:] = sample.cube_pos_w
    sample.goal_pos_w[:] = sample.cube_pos_w + .25 * sample.direction_w
    sample.cube_quat_w[0] = table_rotation
    yaw_45 = math_tools.quaternion_from_rotation_vector(torch.tensor((0., 0., math.pi / 4), dtype=torch.float64))
    sample.cube_quat_w[1] = math_tools.quaternion_multiply(table_rotation, yaw_45)
    tracker.reset(None, sample)
    state = _step(tracker, sample, 1)
    assert state.footprint_failure.tolist() == [False, True]
    assert state.failure.tolist() == [False, True]


def test_fall_and_robot_table_failure_mask_all_positive_rewards_and_beat_success_timeout():
    tracker, sample = _fixture(2)
    _touch(sample)
    sample.cube_pos_w[:] = sample.goal_pos_w
    for counter in range(1, 10):
        _step(tracker, sample, counter)
    sample.robot_table_failure[0] = True
    sample.cube_pos_w[1, 2] = -.011
    env = _environment(tracker, sample, counter=10)
    env.episode_length_buf[:] = 500
    assert terminations.push_failure(env).all()
    assert not terminations.push_success(env).any()
    assert not terminations.push_time_out(env).any()
    assert _step(tracker, sample, 10).fall_failure.tolist() == [False, True]
    for function in (rewards.push_approach_reward, rewards.push_progress_reward, rewards.push_first_contact_reward,
                     rewards.push_contact_reward, rewards.push_success_reward):
        assert not function(env).any()
    torch.testing.assert_close(rewards.push_failure_reward(env) * env.step_dt * 2, torch.full((2,), -2.))


def test_success_masks_exact_timeout_while_plain_timeout_remains_truncation():
    tracker, sample = _fixture(2)
    _touch(sample, rows=0)
    sample.cube_pos_w[0] = sample.goal_pos_w[0]
    for counter in range(1, 11):
        _step(tracker, sample, counter)
    env = _environment(tracker, sample, counter=10)
    env.episode_length_buf[:] = 500
    assert terminations.push_success(env).tolist() == [True, False]
    assert terminations.push_time_out(env).tolist() == [False, True]
    env.episode_length_buf[1] = 499
    assert not terminations.push_time_out(env).any()


def test_subset_reset_invalidates_same_counter_and_masks_stale_contact_without_changing_other_row():
    tracker, sample = _fixture()
    _touch(sample)
    sample.cube_pos_w[:, 1] = .05
    before = _step(tracker, sample, 1)
    sample.cube_pos_w[1] = sample.initial_cube_pos_w[1]
    sample.live[1] = False
    tracker.reset((1,), sample)
    after = _step(tracker, sample, 1)
    assert after.first_contact.tolist() == [True, False]
    assert after.contact_seen.tolist() == [True, False]
    assert after.progress_delta[0] == before.progress_delta[0]
    assert after.progress_delta[1] == 0
    assert before.contact_seen.tolist() == [True, True]  # prior result remains intact
    sample.live[1] = True
    next_state = _step(tracker, sample, 2)
    assert next_state.first_contact.tolist() == [False, True]
    assert not next_state.progress_delta.any()


def test_reward_manager_dt_cancels_only_increment_and_terminal_terms():
    tracker, sample = _fixture()
    _touch(sample)
    sample.cube_pos_w[:, 1] = .002
    env = _environment(tracker, sample)
    integrated_progress = rewards.push_progress_reward(env) * 12 * env.step_dt
    torch.testing.assert_close(integrated_progress, torch.full((2,), .22028441272537863, dtype=torch.float64))
    torch.testing.assert_close(rewards.push_first_contact_reward(env) * .02 * env.step_dt, torch.full((2,), .02))
    torch.testing.assert_close(rewards.push_contact_reward(env) * .005 * env.step_dt,
                               torch.full((2,), .0001, dtype=torch.float64))


def test_bad_layout_raises_and_nonfinite_physics_fails_with_finite_reward_state():
    tracker, sample = _fixture()
    with pytest.raises(ValueError):
        _step(tracker, replace(sample, cube_palm_forces_w_history=torch.zeros(2, 2, 1, 18, 3)), 1)
    with pytest.raises(ValueError):
        _step(tracker, replace(sample, cube_table_forces_w_history=torch.zeros(2, 1, 1, 1, 3)), 1)
    sample.cube_pos_w[0, 0] = float("nan")
    sample.hand_quat_w[1] = 0
    state = _step(tracker, sample, 1)
    assert state.invalid_state.all() and state.failure.all()
    for name in ("approach_delta", "progress_delta", "backslide_delta", "alignment_penalty", "cube_goal_distance_m"):
        assert torch.isfinite(getattr(state, name)).all()
    assert not state.progress_delta.any()


def test_progress_sigma_amplifies_first_millimetre_and_preserves_endpoint_budget():
    tracker, sample = _fixture()
    _touch(sample)
    sample.cube_pos_w[:, 1] = .001
    env = _environment(tracker, sample)
    integrated = rewards.push_progress_reward(env) * 12 * env.step_dt
    torch.testing.assert_close(integrated, torch.full((2,), .11058277283845471, dtype=torch.float64))
    # Previous linear weight 6 paid .024 for 1 mm of a 25 cm command.
    assert (integrated > 4.5 * .024).all()
    assert (integrated < 4.7 * .024).all()
    distances = torch.tensor((.25, .249, 0.), dtype=torch.float64)
    length = torch.full_like(distances, .25)
    broad = state_tools.push_progress_potential(distances, length, .5)
    narrow = state_tools.push_progress_potential(distances, length, .25)
    assert narrow[1] > broad[1]
    torch.testing.assert_close(broad[[0, 2]], torch.tensor((0., 1.), dtype=torch.float64))
    torch.testing.assert_close(narrow[[0, 2]], torch.tensor((0., 1.), dtype=torch.float64))


def test_progress_potential_is_monotonic_bounded_and_scales_with_command_length():
    distance_fraction = torch.tensor((1.1, 1., .8, .5, .2, 0.), dtype=torch.float64)
    short_length = torch.full_like(distance_fraction, .2)
    long_length = torch.full_like(distance_fraction, .3)
    short = state_tools.push_progress_potential(distance_fraction * short_length, short_length, .5)
    long = state_tools.push_progress_potential(distance_fraction * long_length, long_length, .5)
    torch.testing.assert_close(short, long)
    assert short[0] == short[1] == 0
    assert short[-1] == 1
    assert (short[1:] >= short[:-1]).all()
    # Concavity increases early progress sensitivity while retaining a
    # positive increment during the final part of the commanded path.
    assert short[2] > .2
    assert short[-1] > short[-2]


def test_complete_contact_path_budget_stays_one_with_backslide_and_retouch():
    tracker, sample = _fixture(1)
    _touch(sample)
    total = torch.zeros(1, dtype=torch.float64)
    for counter, position in enumerate((.001, .03, .01, .03, .08, .20, .25), start=1):
        sample.cube_pos_w[:, 1] = position
        state = _step(tracker, sample, counter)
        total += state.progress_delta
        if counter == 3:
            torch.testing.assert_close(state.backslide_delta, torch.tensor((-.08,), dtype=torch.float64))
        if counter == 4:
            assert not state.progress_delta.any()  # old record cannot pay twice
    torch.testing.assert_close(total, torch.ones(1, dtype=torch.float64))
    assert not _step(tracker, sample, 8).progress_delta.any()  # camping at goal


def test_initialized_step_cache_and_partial_reset_do_not_read_tensor_booleans(monkeypatch):
    tracker, sample = _fixture()
    _touch(sample)
    sample.cube_pos_w[:, 1] = .001

    def reject_device_boolean(tensor):
        raise AssertionError("initialized tracker must not synchronize a tensor boolean to the host")

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "__bool__", reject_device_boolean)
        first = _step(tracker, sample, 1)
        cached = _step(tracker, sample, 1)
        sample.cube_pos_w[1] = sample.initial_cube_pos_w[1]
        sample.live[1] = False
        tracker.reset((1,), sample)
        reset_state = _step(tracker, sample, 1)
        next_state = _step(tracker, sample, 2)
    assert cached is first
    assert reset_state.progress_delta[0] == first.progress_delta[0]
    assert reset_state.progress_delta[1] == 0
    assert not next_state.progress_delta.any()


@pytest.mark.parametrize("sigma_fraction", (0., -.5, float("nan"), float("inf")))
def test_progress_sigma_rejects_values_that_cannot_define_a_finite_reward(sigma_fraction):
    with pytest.raises(ValueError, match="progress_sigma_fraction"):
        state_tools.PushStateConfig(progress_sigma_fraction=sigma_fraction)
