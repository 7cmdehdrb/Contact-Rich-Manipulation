#!/usr/bin/env python3
"""Actual-physics Push-v1 lift with pre-auto-reset height and reward checks.

Run only after the active GPU fixture finishes. This uses the registered
production environment, assets, reset and controller without overrides.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/hand_manipulation_test'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lift-step-m', type=float, default=.006)
parser.add_argument('--output', type=Path, default=Path(__file__).with_name('height_guard.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.steps <= 100 or args.lift_step_m != .006:
    parser.error('Use 1..100 policy steps and the specified 6mm world-up increment')
launcher = AppLauncher(args, fast_shutdown=True)
app = launcher.app


def save(report):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)


def main():
    import math
    import gymnasium as gym
    import torch
    import hand_manipulation_test
    import isaaclab.utils.math as math_utils
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from hand_manipulation_test.geometry import control_point_pose_w
    from hand_manipulation_test.assets.robot import ROBOT_CONTACT_BODY_NAMES

    task_id = 'Isaac-Hand-Manipulation-Push-v1'
    cfg = load_cfg_from_registry(task_id, 'env_cfg_entry_point')
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    report = {'task': task_id, 'seed': args.seed, 'num_envs': 4,
              'lift_step_m': args.lift_step_m, 'max_steps': args.steps,
              'reward_buffer_semantics': '_step_reward is weighted rate; multiply by policy_dt for contribution',
              'trace': [], 'terminals': [None] * 4, 'passed': False}
    env = gym.make(task_id, cfg=cfg).unwrapped
    try:
        ids = torch.arange(4, device=env.device)
        arm = env.action_manager.get_term('arm_action')
        hand = env.action_manager.get_term('hand_action')
        safe = env.event_manager.get_term_cfg('safe_hand').func
        command = env.command_manager.get_term('target_position')
        env.reset(seed=args.seed)
        # Deterministic normal branches on both sides, preserving production
        # hand randomness, collision certificate, jitter and all reset bounds.
        angle = torch.tensor((0., 0., math.pi, math.pi), device=env.device)
        positions = torch.tensor([
            (cfg.task.table_center[0] + cfg.task.command_midpoint_x_offset_m, y,
             cfg.task.cube_center_height_m) for y in (-.01, .01, -.01, .01)
        ], device=env.device)
        command.set_reset_specs(ids, positions, angle, .20)
        env.reset(seed=args.seed)
        assert bool(safe.check_final(ids).all()), 'Initial whole-robot reset certificate failed'
        assert cfg.task.eef_soft_height_above_table_m == .15
        assert cfg.task.eef_hard_height_above_table_m == .25
        assert cfg.rewards.eef_height.weight == 5.
        assert abs(env.step_dt - .02) < 1.e-10
        assert arm.cfg.position_error_limit_m == .06
        report.update(policy_dt=env.step_dt, physics_dt=env.physics_dt,
                      soft_height_m=.15, hard_height_m=.25, height_weight=5.,
                      position_target_error_limit_m=.06,
                      initial_hand_synergy=safe.initial_hand_synergy.cpu().tolist(),
                      initial_arm_q=env.scene['robot'].data.joint_pos[:, safe.arm_joint_ids].cpu().tolist(),
                      robot_table_contact_names=ROBOT_CONTACT_BODY_NAMES)
        names = tuple(env.reward_manager.active_terms)
        height_index = names.index('eef_height')
        positive_names = ('approach', 'progress', 'first_contact', 'contact', 'success')
        positive_indices = [names.index(name) for name in positive_names]
        report['reward_names'] = names
        completed = torch.zeros(4, device=env.device, dtype=torch.bool)
        positive_steps = torch.zeros(4, device=env.device, dtype=torch.long)
        positive_above_soft_steps = torch.zeros_like(positive_steps)
        processed_error_norm = torch.zeros(4, device=env.device)
        original_process = arm.process_actions_for_envs

        def process(actions, env_ids):
            original_process(actions, env_ids)
            processed_error_norm.copy_(arm.position_target_error_c_m.norm(dim=-1))

        arm.process_actions_for_envs = process
        original_record = command._record_metrics
        last_capture_counter = env._sim_step_counter
        captured = None
        step = -1

        def record(state):
            nonlocal last_capture_counter, captured
            original_record(state)
            counter = env._sim_step_counter
            if counter == last_capture_counter:
                return
            # Capture the first physical transition at this counter. Later
            # reset/command/observation updates at the same counter cannot
            # overwrite the terminal episode's pose with its fresh reset.
            last_capture_counter = counter
            position, _ = control_point_pose_w(env)
            table = env.scene['table'].data
            local = math_utils.quat_apply_inverse(table.root_quat_w, position-table.root_pos_w)
            height = local[:, 2]-cfg.task.table_size[2]/2
            torch.testing.assert_close(height, command._eef_height, rtol=0., atol=2.e-6)
            captured = {'step': step, 'physics_counter': counter,
                        'actual_c_pos_w': position.cpu().tolist(),
                        'actual_table_pos_w': table.root_pos_w.cpu().tolist(),
                        'height_above_table_m': height.cpu().tolist(),
                        'failure': state.failure.cpu().tolist(), 'success': state.success.cpu().tolist(),
                        'height_failure': command._height_failure.cpu().tolist(),
                        'finger_failure': command._finger_failure.cpu().tolist(),
                        'wrist_failure': command._wrist_failure.cpu().tolist(),
                        'finger_outward_cos': command._finger_outward_cos.cpu().tolist(),
                        'wrist_2_angle_rad': command._wrist_angle.cpu().tolist(),
                        'table_failure': state.table_failure.cpu().tolist(),
                        'footprint_failure': state.footprint_failure.cpu().tolist(),
                        'fall_failure': state.fall_failure.cpu().tolist(),
                        'invalid_state': state.invalid_state.cpu().tolist(),
                        'additional_failure': state.additional_failure.cpu().tolist(),
                        'robot_table_force_n': env.scene['table_contacts'].data.force_matrix_w_history[:,:,0].norm(dim=-1).amax(1).cpu().tolist()}

        command._record_metrics = record
        save(report)
        try:
            for step in range(args.steps):
                active = ~completed
                _, current_q = control_point_pose_w(env)
                up_world = torch.zeros(4, 3, device=env.device)
                up_world[:, 2] = args.lift_step_m
                delta_c = math_utils.quat_apply_inverse(current_q, up_world)
                action = torch.zeros(4, 8, device=env.device)
                action[:, :3] = delta_c / action.new_tensor(arm.cfg.translation_scale)
                action[completed, :6] = 0.
                # Selected rows keep the originally requested physical hand;
                # newly reset completed rows hold their new safe reset target.
                action[:, 6:] = hand.synergy_to_action(safe.initial_hand_synergy)
                assert bool((action.abs() <= 1.+1.e-7).all()), 'Fixture exceeds the physical action range'
                captured = None
                _, reward, terminated, truncated, _ = env.step(action)
                assert captured is not None, 'No pre-auto-reset physical state was captured'
                assert bool((processed_error_norm[active] <= .06+2.e-6).all()), 'Processed translation target exceeds its norm cap'
                # RewardManager.reset leaves this buffer intact. It stores
                # weighted rates, so multiplying by dt recovers each actual
                # contribution from the terminal episode before auto-reset.
                contributions = env.reward_manager._step_reward.clone()*env.step_dt
                height = contributions.new_tensor(captured['height_above_table_m'])
                expected = -5.*(height-.15).clamp_min(0.)*.02
                torch.testing.assert_close(contributions[active,height_index], expected[active], rtol=0., atol=2.e-6)
                positive = (contributions[:,positive_indices] > 1.e-8).any(-1) & active
                positive_steps += positive.long()
                positive_above_soft_steps += (positive & (height >= .15)).long()
                captured.update(active=active.cpu().tolist(), terminated=terminated.cpu().tolist(),
                                truncated=truncated.cpu().tolist(),
                                processed_position_error_norm_m=processed_error_norm.cpu().tolist(),
                                weighted_height_contribution=contributions[:,height_index].cpu().tolist(),
                                expected_height_contribution=expected.cpu().tolist(),
                                reward_contributions=contributions.cpu().tolist(), total_reward=reward.cpu().tolist())
                report['trace'].append(captured)
                ended = (terminated | truncated) & active
                for i in ids[ended].cpu().tolist():
                    terminal = {name: value[i] for name,value in captured.items()
                                if isinstance(value,list) and len(value)==4}
                    terminal['step'] = step
                    report['terminals'][i] = terminal
                completed |= ended
                save(report)
                if bool(completed.all()):
                    break
            report['positive_reward_steps_below_soft_or_above'] = positive_steps.cpu().tolist()
            report['positive_reward_steps_above_soft'] = positive_above_soft_steps.cpu().tolist()
            assert bool(completed.all()), 'Some original rows did not terminate within100 lift steps'
            for i, terminal in enumerate(report['terminals']):
                assert terminal['terminated'] and not terminal['truncated'], f'Row{i} did not return a real hard termination'
                assert terminal['height_above_table_m'] >= .25, f'Row{i} terminated below the actual hard height'
                assert terminal['height_failure'] and terminal['additional_failure'] and terminal['failure']
                for cause in ('table_failure','footprint_failure','fall_failure','invalid_state','finger_failure','wrist_failure','success'):
                    assert not terminal[cause], f'Row{i} has an additional failure cause:{cause}'
                rewards = terminal['reward_contributions']
                assert all(abs(rewards[j]) <= 1.e-8 for j in positive_indices), f'Row{i} collected a positive task reward at height failure'
            report['passed'] = True
        finally:
            command._record_metrics = original_record
            arm.process_actions_for_envs = original_process
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        save(report)
        env.close()
    print('HEIGHT_GUARD_RESULT', json.dumps({name:report[name] for name in
          ('passed','positive_reward_steps_below_soft_or_above','positive_reward_steps_above_soft','terminals')}), flush=True)

status = 0
try:
    main()
except Exception:
    traceback.print_exc()
    status = 1
finally:
    import omni.kit.app
    sys.stdout.flush()
    omni.kit.app.get_app().post_quit(status)
    app.close()
raise SystemExit(status)
