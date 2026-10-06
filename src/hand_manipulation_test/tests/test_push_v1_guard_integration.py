"""Exercise production v1 command guards before physical reward bookkeeping."""

from contextlib import contextmanager
from dataclasses import fields
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_push_commands import _realize_reset
from test_push_terms import rewards as push_rewards, terminations
from test_push_v1_rewards import _fixture


@contextmanager
def _guard_fixture(count=2, *, enforce=False):
    env, original, rewards, math_tools, safe = _fixture(count, height=.025, preferred_roll=0.)
    task = env.cfg.task
    task.eef_soft_height_above_table_m = .15
    task.eef_hard_height_above_table_m = .25
    task.enforce_outward_fingers = enforce
    task.minimum_finger_outward_cos = .25
    task.wrist_2_branch_sin_margin = .15
    env.scene['robot'] = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.full((count, 1), math.pi/2)),
                                         find_joints=lambda name, preserve_order: ([0], ['wrist_2_joint']))
    env.control_position = env.eef_position.clone()
    production_globals = original.__class__._snapshot.__globals__
    with patch.dict(production_globals, {'control_point_pose_w': lambda e: (e.control_position, e.eef_quaternion)}):
        command = original.__class__(original.cfg, env)
        env.command_manager = SimpleNamespace(get_term=lambda _: command)
        env.episode_length_buf.zero_()
        command.set_reset_specs(torch.arange(count), [-.70, 0., 1.08], 0., .25)
        _realize_reset(env, command, torch.arange(count))
        yield env, command, rewards, math_tools, safe


def _step(env, command, counter):
    env.episode_length_buf[:] = 1
    env._sim_step_counter = counter
    return command.state()


def _touch(env):
    env.scene['cube_palm_contacts'].data.force_matrix_w_history[:, :, 0, 0, 1] = .02


def test_optional_actual_contact_reference_changes_approach_only_and_keeps_cube_goal_and_c_height():
    with _guard_fixture() as (env, command, rewards, _, safe):
        assert not hasattr(safe, "contact_reference_pose_w")
        initial, goal = command.target_pos_w.clone(), command.goal_pos_w.clone()
        central = env.eef_position.clone()
        reference = torch.tensor([[-.70, -.13, 1.105], [-.70, -.13, 1.105]])
        safe.contact_reference_pose_w = lambda: (reference, env.eef_quaternion)
        first = _step(env, command, 2)
        assert first.approach_delta.min() > 0
        assert not first.progress_delta.any() and not first.valid_push_contact.any()
        torch.testing.assert_close(command.metrics["approach_gap_m"], torch.full((2,), .10), atol=3.e-7, rtol=0.)
        torch.testing.assert_close(rewards.push_v1_contact_distance_penalty(env), torch.full((2,), -.10), atol=3.e-7, rtol=0.)
        torch.testing.assert_close(command._latest_snapshot.palm_pos_w, reference)
        torch.testing.assert_close(command._latest_snapshot.approach_target_pos_w,
                                   torch.tensor([[-.70, -.03, 1.105], [-.70, -.03, 1.105]]), atol=2.e-7, rtol=0.)
        reference[:, 1] = -.03
        next_state = _step(env, command, 4)
        assert next_state.approach_delta.min() > 0
        assert rewards.push_v1_contact_distance_penalty(env).abs().max() < 2.e-7
        torch.testing.assert_close(command.metrics["eef_height_above_table_m"], torch.full((2,), .13), atol=2.e-7, rtol=0.)
        torch.testing.assert_close(command.target_pos_w, initial, atol=0., rtol=0.)
        torch.testing.assert_close(command.goal_pos_w, goal, atol=0., rtol=0.)
        torch.testing.assert_close(env.eef_position, central, atol=0., rtol=0.)
        assert not next_state.progress_delta.any() and not next_state.contact_seen.any()


def test_contact_reference_fallback_preserves_old_fixture_and_partial_reset_record():
    with _guard_fixture() as (env, command, rewards, _, safe):
        env.eef_position[:] = torch.tensor([-.70, -.03, 1.105])
        assert _step(env, command, 2).approach_delta.min() > 0
        assert command._latest_snapshot.approach_target_pos_w is None
        assert rewards.push_v1_contact_distance_penalty(env).abs().max() < 2.e-7
        reference = env.eef_position.clone()
        safe.contact_reference_pose_w = lambda: (reference, env.eef_quaternion)
        _step(env, command, 4)
        saved_goal, saved_initial = command.goal_pos_w[1].clone(), command.target_pos_w[1].clone()
        saved_record = command.tracker._best_approach[1].clone()
        reference[0, 1] = -.13
        env.episode_length_buf[0] = 0
        command.set_reset_specs([0], [-.70, .02, 1.08], 0., .25)
        _realize_reset(env, command, torch.tensor([0]))
        state = command.state()
        assert state.approach_delta[0] == 0 and rewards.push_v1_contact_distance_penalty(env)[0] == 0
        torch.testing.assert_close(command.tracker._best_approach[1], saved_record, atol=0., rtol=0.)
        torch.testing.assert_close(command.goal_pos_w[1], saved_goal, atol=0., rtol=0.)
        torch.testing.assert_close(command.target_pos_w[1], saved_initial, atol=0., rtol=0.)
        assert not state.approach_delta[1]
        # Repeated readers at the same physics counter retain the same record.
        assert command.state() is state


def test_actual_control_point_height_is_independent_from_safe_palm_and_contact_target():
    with _guard_fixture(3) as (env, command, rewards, _, _):
        # Palm stays on the ordinary contact plane in every row. Only the
        # actual massless C frame changes; wrist/hand origins are not a proxy.
        env.eef_position[:] = torch.tensor([-.70, -.03, 1.105])
        env.control_position[:, 2] = torch.tensor([1.18, 1.25, 1.3001])
        state = _step(env, command, 2)
        torch.testing.assert_close(command.metrics['palm_height_error_m'], torch.zeros(3), atol=2e-7, rtol=0.)
        torch.testing.assert_close(command.metrics['eef_height_above_table_m'], torch.tensor([.13, .20, .2501]), atol=2e-7, rtol=0.)
        torch.testing.assert_close(rewards.push_v1_eef_height_penalty(env), torch.tensor([0., -.05, -.1001]), atol=2e-7, rtol=0.)
        assert state.additional_failure.tolist() == state.failure.tolist() == [False, False, True]
        assert command.metrics['eef_height_failure'].tolist() == [0., 0., 1.]
        assert not command.metrics['table_failure'].any()
        assert not command.metrics['wrist_branch_failure'].any()
        assert not command.metrics['finger_inward_failure'].any()


def test_height_failure_beats_success_and_timeout_and_zeroes_all_positive_terms():
    with _guard_fixture(3) as (env, command, _, _, _):
        _touch(env)
        env.scene['target_object'].data.root_pos_w[:2] = command.goal_pos_w[:2]
        for counter in range(2, 20, 2):
            assert not _step(env, command, counter).success.any()
        env.control_position[0, 2] = 1.3001
        env.episode_length_buf[:] = 500
        env._sim_step_counter = 20
        assert terminations.push_failure(env).tolist() == [True, False, False]
        assert terminations.push_success(env).tolist() == [False, True, False]
        assert terminations.push_time_out(env).tolist() == [False, False, True]
        for term in (push_rewards.push_approach_reward, push_rewards.push_progress_reward,
                     push_rewards.push_first_contact_reward, push_rewards.push_contact_reward,
                     push_rewards.push_success_reward):
            assert term(env)[0] == 0, term.__name__
        assert push_rewards.push_failure_reward(env)[0] == -50
        assert command.metrics['eef_height_failure'].tolist() == [1., 0., 0.]
        assert command.metrics['table_failure'].sum() == 0


def test_contact_distance_remains_negative_after_record_recession_and_reacquisition():
    with _guard_fixture() as (env, command, rewards, _, _):
        env.eef_position[:] = torch.tensor([-.70, -.03, 1.105])
        assert _step(env, command, 2).approach_delta.min() > 0
        assert rewards.push_v1_contact_distance_penalty(env).abs().max() < 2e-7
        env.eef_position[:, 1] = -.13
        assert not _step(env, command, 4).approach_delta.any()
        torch.testing.assert_close(rewards.push_v1_contact_distance_penalty(env), torch.full((2,), -.10), atol=2e-7, rtol=0.)
        _touch(env)
        env.eef_position[:, 1] = -.08
        assert _step(env, command, 6).first_contact.all()
        torch.testing.assert_close(rewards.push_v1_contact_distance_penalty(env), torch.full((2,), -.05), atol=2e-7, rtol=0.)
        env.scene['cube_palm_contacts'].data.force_matrix_w_history.zero_()
        env.eef_position[:, 1] = -.13
        assert not _step(env, command, 8).approach_delta.any()
        torch.testing.assert_close(rewards.push_v1_contact_distance_penalty(env), torch.full((2,), -.10), atol=2e-7, rtol=0.)
        _touch(env)
        env.eef_position[:, 1] = -.03
        assert not _step(env, command, 10).first_contact.any()
        assert rewards.push_v1_contact_distance_penalty(env).abs().max() < 2e-7
        assert (rewards.push_v1_contact_distance_penalty(env) <= 0).all()


def test_new_negative_rates_apply_weight_and_manager_dt_once_and_zero_reset_rows():
    with _guard_fixture() as (env, command, rewards, _, _):
        env.control_position[:, 2] = 1.25
        env.eef_position[:] = torch.tensor([-.70, -.13, 1.105])
        _step(env, command, 2)
        height_rate = rewards.push_v1_eef_height_penalty(env).clone()
        distance_rate = rewards.push_v1_contact_distance_penalty(env).clone()
        torch.testing.assert_close(height_rate*5*.02, torch.full((2,), -.005), atol=2e-8, rtol=0.)
        torch.testing.assert_close(distance_rate*1*.02, torch.full((2,), -.002), atol=2e-8, rtol=0.)
        env.step_dt = .01
        torch.testing.assert_close(rewards.push_v1_eef_height_penalty(env), height_rate)
        torch.testing.assert_close(rewards.push_v1_contact_distance_penalty(env), distance_rate)
        env.episode_length_buf[1] = 0
        env._sim_step_counter = 4
        command.state()
        assert rewards.push_v1_eef_height_penalty(env)[1] == 0
        assert rewards.push_v1_contact_distance_penalty(env)[1] == 0
        assert command.metrics['eef_height_above_table_m'][1] == 0


def test_wrapped_positive_wrist_branch_passes_but_near_singularity_negative_and_inward_fail():
    with _guard_fixture(5, enforce=True) as (env, command, _, math_tools, _):
        env.scene['robot'].data.joint_pos[:, 0] = torch.tensor([math.pi/2, math.pi/2+2*math.pi, .01, -.5, math.pi/2])
        env.eef_quaternion[4] = math_tools.quaternion_from_rotation_vector(torch.tensor([0., math.pi/2, 0.]))
        _touch(env)
        env.scene['target_object'].data.root_pos_w[:, 1] += .01
        state = _step(env, command, 2)
        assert state.failure.tolist() == state.additional_failure.tolist() == [False, False, True, True, True]
        assert command.metrics['wrist_branch_failure'].tolist() == [0., 0., 1., 1., 0.]
        assert command.metrics['finger_inward_failure'].tolist() == [0., 0., 0., 0., 1.]
        assert command.metrics['eef_height_failure'].sum() == 0
        assert state.first_contact.tolist() == [True, True, False, False, False]
        assert bool((state.progress_delta[:2] > 0).all()) and not state.progress_delta[2:].any()
        torch.testing.assert_close(command.metrics['wrist_2_angle_rad'][:2], torch.full((2,), math.pi/2), atol=4e-7, rtol=0.)


def test_partial_reset_clears_only_selected_guard_causes_and_keeps_cache_histories():
    with _guard_fixture(3, enforce=True) as (env, command, rewards, _, _):
        _touch(env)
        env.control_position[0, 2] = 1.3001
        env.scene['robot'].data.joint_pos[1, 0] = -.5
        before = _step(env, command, 2)
        old_metrics = {name: value.clone() for name, value in command.metrics.items()}
        with patch.object(command, '_snapshot', wraps=command._snapshot) as snapshot:
            for _ in range(3):
                command.compute(.02)
                rewards.push_v1_contact_distance_penalty(env)
                rewards.push_v1_eef_height_penalty(env)
                assert command.state() is before
            assert snapshot.call_count == 0
        torch.testing.assert_close(command.metrics['raw_contact_time_s'], torch.full((3,), .02))
        assert before.failure.tolist() == [True, True, False]
        command.set_reset_specs([0], [-.70, 0., 1.08], 0., .25)
        env.control_position[0, 2] = 1.18
        _realize_reset(env, command, [0])
        env.episode_length_buf[0] = 0
        after = command.state()
        assert after.failure.tolist() == after.additional_failure.tolist() == [False, True, False]
        for item in fields(before):
            torch.testing.assert_close(getattr(after, item.name)[1:], getattr(before, item.name)[1:])
        for name in ('eef_height_failure', 'wrist_branch_failure', 'raw_contact_time_s', 'valid_push_contact_time_s'):
            assert command.metrics[name][0] == 0
            torch.testing.assert_close(command.metrics[name][1:], old_metrics[name][1:])
        assert rewards.push_v1_eef_height_penalty(env)[0] == 0
        assert rewards.push_v1_contact_distance_penalty(env)[0] == 0
        assert command.state() is after
