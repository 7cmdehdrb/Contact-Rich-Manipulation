"""Full configuration comparison and GPU reset/rollout checks for IK-v0_2-Test."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--num-envs', type=int, default=32)
parser.add_argument('--output', type=Path, default=Path('outputs/relative_ik_v0_2_test_validation'))
parser.add_argument('--config-only', action='store_true')
parser.add_argument('--child', action='store_true')
parser.add_argument('--benchmark', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline
    from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorRelativeIKV0_1EnvCfg
    from inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs import WrenchRefactorRelativeIKV0_2TestEnvCfg
    from inspire_sweep_baseline.runtime.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestEnv
    from inspire_sweep_baseline.mdp.rewards.contact_sweep_clutter_fusion_holding import FusionHoldingV2RewardsCfg
    from inspire_sweep_baseline.mdp.curricula.contact_sweep_clutter_fusion_holding import FusionHoldingV2CurriculumCfg
    from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import WrenchRefactorRelativeIKV0_1PPORunnerCfg
    from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_holding import target_measurements
    task = 'Inspire-Wrench-Refactor-Relative-IK-v0_2-Test'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    parent = WrenchRefactorRelativeIKV0_1EnvCfg().to_dict()
    current = cfg.to_dict()
    expected = copy.deepcopy(parent)
    expected['rewards'] = FusionHoldingV2RewardsCfg().to_dict()
    expected['rewards'].pop('arm_speed_excess')
    expected['rewards']['goal_hold']['params']['position_tolerance'] = .03
    expected['rewards']['shelf_failure']['func'] = current['rewards']['shelf_failure']['func']
    expected['terminations'].pop('target_goal_success')
    expected['terminations']['timeout']['func'] = current['terminations']['timeout']['func']
    schedules = FusionHoldingV2CurriculumCfg().to_dict()
    expected['curriculum'] = {k:v for k,v in schedules.items() if k not in (
        'obstacle_population','goal_deadzone','goal_position_scale','goal_position_tolerance','arm_free_speed')}
    expected['curriculum'].update(parent['curriculum'])
    assert current == expected, [(key, current[key],expected.get(key)) for key in current if current[key] != expected.get(key)]
    assert WrenchRefactorRelativeIKV0_1EnvCfg().to_dict() == parent
    assert WrenchRefactorRelativeIKV0_2TestEnvCfg.__bases__ == (ManagerBasedRLEnvCfg,)
    assert WrenchRefactorRelativeIKV0_2TestEnv.__bases__ == (ManagerBasedRLEnv,)
    runner = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    expected_runner = WrenchRefactorRelativeIKV0_1PPORunnerCfg().to_dict()
    expected_runner['experiment_name'] = 'inspire_wrench_refactor_relative_ik_v0_2-Test'
    assert runner.to_dict() == expected_runner
    if args.child:
        task = 'Inspire-Wrench-Refactor-Relative-IK-v0_2_1-Test'
        child = load_cfg_from_registry(task, 'env_cfg_entry_point')
        child_dict = child.to_dict()
        expected_child = copy.deepcopy(current)
        expected_child['rewards'].update(object=None, eef_linear_speed=None, pushing_speed=None)
        for name in ('object_tilt', 'obstacle_displacement'):
            expected_child['rewards'][name] = child_dict['rewards'][name]
        expected_child['curriculum'].update(eef_free_speed=None, pushing_free_speed=None)
        assert child_dict == expected_child
        assert type(child).__bases__ == (WrenchRefactorRelativeIKV0_2TestEnvCfg,)
        assert WrenchRefactorRelativeIKV0_2TestEnvCfg().to_dict() == current
        child_runner = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point').to_dict()
        expected_runner['experiment_name'] = 'inspire_wrench_refactor_relative_ik_v0_2_1-Test'
        assert child_runner == expected_runner
        cfg = child
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'task':task, 'full_config_comparison':True, 'ppo_comparison':True, 'parent_preserved':True}
    print('FULL CONFIG COMPARISON PASSED', flush=True)
    if not args.config_only:
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = 42
        cfg.log_dir = str(args.output)
        env = gym.make(task,cfg=cfg)
        raw = env.unwrapped
        obs,_ = env.reset(seed=42)
        assert set(obs) == {'policy','obstacles'}
        assert obs['policy'].shape == (args.num_envs,58)
        assert obs['obstacles'].shape == (args.num_envs,64)
        assert raw.action_manager.total_action_dim == 8
        assert raw.termination_manager.active_terms == ['object_fall_or_tilt','shelf_failure','arm_joint_velocity','timeout']
        # Verify the actual PhysX limits, not just serialized configuration.
        from inspire_sweep_baseline.assets.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES
        robot = raw.scene['robot']
        arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        hand_ids = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)[0]
        effort = robot.root_physx_view.get_dof_max_forces()
        velocity = robot.root_physx_view.get_dof_max_velocities()
        torch.testing.assert_close(effort[:,arm_ids], effort.new_tensor([150.,150.,150.,28.,28.,28.]).expand(args.num_envs,-1))
        torch.testing.assert_close(velocity[:,arm_ids], torch.full_like(velocity[:,arm_ids],3.14))
        torch.testing.assert_close(effort[:,hand_ids], torch.full_like(effort[:,hand_ids],10.))
        torch.testing.assert_close(velocity[:,hand_ids], torch.full_like(velocity[:,hand_ids],1.))
        assert 'arm_speed_excess' not in raw.reward_manager.active_terms
        assert 'arm_free_speed' not in raw.curriculum_manager.active_terms
        result['physics_joint_limits_verified'] = True
        result['active_rewards'] = raw.reward_manager.active_terms
        result['active_terminations'] = raw.termination_manager.active_terms
        result['active_curricula'] = raw.curriculum_manager.active_terms
        result['target_root_z_range'] = [float(raw.scene['object_collection'].data.object_pos_w[:,0,2].min()),float(raw.scene['object_collection'].data.object_pos_w[:,0,2].max())]
        assert target_measurements(raw)[2].all(), 'Holding target validity is incompatible with spawned target'
        done = 0
        with torch.inference_mode():
            for iteration in (0,1250,6000,12000,25000,0):
                raw.set_learning_iteration(iteration)
                assert raw.reward_manager.get_term_cfg('goal_hold').params['position_tolerance'] == .03
                for _ in range(20):
                    action = torch.zeros(args.num_envs,8,device=raw.device)
                    action[:,6:] = 1.
                    obs,reward,terminated,truncated,_ = env.step(action)
                    assert torch.isfinite(reward).all()
                    assert all(torch.isfinite(v).all() for v in obs.values())
                    assert (obs['policy'][:,-6:].abs() <= obs['policy'].new_tensor([150.,150.,150.,8.,8.,8.])+1e-5).all()
                    done += int((terminated | truncated).sum())
            assert (raw._metric_counts > 0).all(), 'metrics not collected without success termination'
            if args.child:
                assert all(n not in raw.reward_manager.active_terms for n in ('object','eef_linear_speed','pushing_speed','arm_speed_excess'))
                from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion import combined_active_object_cost
                terms = [raw.reward_manager.get_term_cfg(n) for n in ('object_tilt','obstacle_displacement')]
                split = sum(t.weight * t.func(raw, **t.params) for t in terms)
                torch.testing.assert_close(split, -30 * combined_active_object_cost(raw, displacement_ratio=5.))
            if args.benchmark:
                import time, statistics
                collect = raw.record_goal_accuracy_metrics
                timings = {'enabled': [], 'disabled': []}
                # Alternating equal-length blocks on the same scene; include reset costs.
                for enabled in (False, True, True, False, False, True, True, False):
                    raw.record_goal_accuracy_metrics = collect if enabled else lambda: None
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(40):
                        env.step(action)
                    torch.cuda.synchronize()
                    timings['enabled' if enabled else 'disabled'].append((time.perf_counter()-start)/40)
                raw.record_goal_accuracy_metrics = collect
                result['step_benchmark_seconds'] = timings
                result['step_benchmark_overhead_percent'] = 100 * (
                    statistics.median(timings['enabled'])/statistics.median(timings['disabled'])-1)
            before_counts = raw._metric_counts.clone()
            ids = torch.arange(0,args.num_envs,2,device=raw.device)
            before = raw.scene['object_collection'].data.object_pos_w.clone()
            raw._reset_idx(ids)
            mask = torch.ones(args.num_envs,dtype=torch.bool,device=raw.device); mask[ids] = False
            torch.testing.assert_close(raw.scene['object_collection'].data.object_pos_w[mask],before[mask])
            assert not raw.sensor_data_fresh[ids].any()
            assert not raw._metric_counts[ids].any()
            torch.testing.assert_close(raw._metric_counts[mask], before_counts[mask])
            logs = {k:v for k,v in raw.extras['log'].items() if k.startswith('Metrics/')}
            assert len(logs) == 8, logs
            assert all(torch.isfinite(v) for v in logs.values())
            assert not any('obstacle_actual' in k or 'layout_complete' in k for k in logs)
            result['metrics'] = {k:float(v) for k,v in logs.items()}
            # Force an actual timeout; metrics must contain pre-reset terminal distance.
            terminal_distance = (raw.scene['object_collection'].data.object_pos_w[:,0,:2] - raw.command_manager.get_command('target_goal_pos')[:,:2]).norm(dim=-1)
            raw.episode_length_buf[:] = raw.max_episode_length
            assert raw.termination_manager.compute().all()
            torch.testing.assert_close(raw._metric_final_distance, terminal_distance)

            # Place stationary targets at goal, evaluate after physics sensor refresh.
            states = raw.scene['object_collection'].data.object_link_state_w.clone()
            goals = raw.command_manager.get_command('target_goal_pos')
            states[:,0,:2] = goals[:,:2]
            states[:,0,7:] = 0.
            raw.scene['object_collection'].write_object_link_state_to_sim(states)
            raw.scene.write_data_to_sim()
            raw.sim.forward()
            raw.scene.update(raw.step_dt)
            hold_cfg = raw.reward_manager.get_term_cfg('goal_hold')
            hold = hold_cfg.func(raw,**hold_cfg.params)
            assert (hold > .99).all(), hold
            # No cumulative 3-second hold budget or success termination remains.
            for _ in range(151):
                assert (hold_cfg.func(raw,**hold_cfg.params) > .99).all()
            result.update(num_envs=args.num_envs,rollout_steps=120,done_count=done,
                          partial_reset=True,goal_hold_at_rest=True,hold_over_3s_uncapped=True)
        env.close()
    (args.output / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('IK V0_2 TEST PROBE PASSED',flush=True)
except BaseException:
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
finally:
    app.close()
