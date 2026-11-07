"""Real config allowlist and GPU spawn/reset/observation checks for Relative v0/v2/v3."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=int, choices=(0, 2, 3), default=0)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--output', type=Path, default=Path('outputs/wrench_refactor_relative_validation'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorV3_1EnvCfg, WrenchRefactorRelativeEnvCfg
    from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import WrenchRefactorV3_1PPORunnerCfg, WrenchRefactorRelativePPORunnerCfg
    from inspire_sweep_baseline.mdp.action_functions.wrench_refactor import WrenchRefactorOSCAction
    from inspire_sweep_baseline.mdp.reward_functions.wrench_refactor import policy_action_rate_penalty

    parent = WrenchRefactorV3_1EnvCfg()
    original = copy.deepcopy(parent.to_dict())
    cfg = WrenchRefactorRelativeEnvCfg()
    expected = copy.deepcopy(original)
    expected['observations'] = cfg.observations.to_dict()
    expected['scene']['robot']['actuators']['hand']['effort_limit_sim'] = 10.
    expected['scene']['robot']['spawn']['sensor_mass'] = .280
    expected['rewards']['target_goal_first_entry_bonus']['weight'] = 10.
    expected['rewards']['target_goal_distance'] = cfg.rewards.target_goal_distance.to_dict()
    expected['rewards']['target_inverse_distance'] = cfg.rewards.target_inverse_distance.to_dict()
    expected['wrist_wrench_observation_limits'] = (150.,150.,150.,8.,8.,8.)
    expected['rewards']['robot_action_rate_penalty']['params'] = {'arm_max_cost':100., 'hand_max_cost':100.}
    expected['scene']['robot']['actuators']['arm']['velocity_limit_sim'] = dict.fromkeys(cfg.scene.robot.actuators['arm'].velocity_limit_sim, 3.14)
    expected['scene']['robot']['actuators']['arm']['effort_limit_sim'] = dict(zip(cfg.scene.robot.actuators['arm'].effort_limit_sim, (150.,150.,150.,28.,28.,28.)))
    expected['terminations']['arm_joint_velocity']['params']['threshold'] = 4.0
    expected['curriculum']['arm_joint_velocity_threshold'] = cfg.curriculum.arm_joint_velocity_threshold.to_dict()
    assert cfg.to_dict() == expected, 'Unexpected environment delta'
    assert parent.to_dict() == original == WrenchRefactorV3_1EnvCfg().to_dict()
    assert cfg.actions.arm_action.class_type is WrenchRefactorOSCAction
    assert cfg.rewards.robot_action_rate_penalty.func is policy_action_rate_penalty
    # Inspect every observation term rather than allowing arbitrary changes in the group.
    obs_expected = copy.deepcopy(original['observations'])
    policy = obs_expected['policy']
    for name in ('target_obs_state', 'goal_pos', 'tactile'):
        policy[name]['func'] = cfg.observations.policy.to_dict()[name]['func']
    policy['target_obs_state']['noise'] = None
    policy['target_obs_state']['params']['coordinate_noise'] = .01
    policy['tactile']['noise'] = cfg.observations.policy.tactile.to_dict()['noise']
    obs_expected['obstacles']['initial_positions']['func'] = cfg.observations.obstacles.initial_positions.to_dict()['func']
    obs_expected['critic'] = obs_expected['critic_obstacles'] = None
    assert cfg.observations.to_dict() == obs_expected
    runner = WrenchRefactorRelativePPORunnerCfg()
    runner_expected = WrenchRefactorV3_1PPORunnerCfg().to_dict()
    runner_expected['experiment_name'] = 'inspire_wrench_refactor_relative_v0'
    runner_expected['obs_groups']['critic'] = ['policy', 'obstacles']
    for name in ('actor', 'critic'):
        runner_expected[name]['class_name'] = runner.to_dict()[name]['class_name']
    runner_expected['critic']['obstacle_group'] = 'obstacles'
    assert runner.to_dict() == runner_expected, 'Unexpected PPO delta'
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorRelativeV2EnvCfg, WrenchRefactorRelativeV3EnvCfg
    from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import WrenchRefactorRelativeV2PPORunnerCfg, WrenchRefactorRelativeV3PPORunnerCfg
    assert 'Inspire-Wrench-Refactor-Relative-v1' not in gym.registry
    v0_snapshot = copy.deepcopy(cfg.to_dict())
    v3 = WrenchRefactorRelativeV3EnvCfg()
    v3_expected = copy.deepcopy(v0_snapshot)
    v3_expected['rewards']['target_contact'] = None
    v3_expected['rewards']['target_goal_distance'] = v3.rewards.target_goal_distance.to_dict()
    for name in ('target_goal_first_entry_bonus', 'target_goal_hold', 'target_goal_success_bonus'):
        v3_expected['rewards'][name] = None
    v3_expected['terminations']['target_goal_success'] = None
    assert v3.to_dict() == v3_expected
    assert cfg.to_dict() == v0_snapshot == WrenchRefactorRelativeEnvCfg().to_dict()
    assert WrenchRefactorRelativeV3EnvCfg.__bases__ == (WrenchRefactorRelativeEnvCfg,)
    v3_runner = WrenchRefactorRelativeV3PPORunnerCfg()
    v3_runner_expected = runner.to_dict()
    v3_runner_expected['experiment_name'] = 'inspire_wrench_refactor_relative_v3'
    assert v3_runner.to_dict() == v3_runner_expected
    if args.version == 2:
        cfg, runner = WrenchRefactorRelativeV2EnvCfg(), WrenchRefactorRelativeV2PPORunnerCfg()
    elif args.version == 3:
        cfg, runner = v3, v3_runner
    print('CONFIG_ALLOWLIST_AND_PARENT_ISOLATION_OK', flush=True)

    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 42
    cfg.log_dir = str(args.output)
    env = gym.make(f'Inspire-Wrench-Refactor-Relative-v{args.version}', cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    expected_dims = {'policy':58, 'obstacles':64}
    if args.version == 2:
        expected_dims['critic'] = 58
    assert {k:v.shape[-1] for k,v in obs.items()} == expected_dims
    # Exercise the real manager's post-noise clipping, including corruption-off.
    for group in ('policy', 'critic') if args.version == 2 else ('policy',):
        term = getattr(raw.observation_manager.cfg, group).wrist_wrench
        reader, noise = term.func, term.noise
        sample = torch.tensor([[200.,-200.,150.,10.,-10.,8.]], device=raw.device).repeat(raw.num_envs,1)
        try:
            term.func = lambda env: sample.clone()
            term.noise = None
            expected = sample.clamp(min=term.clip[0], max=term.clip[1])
            torch.testing.assert_close(raw.observation_manager.compute_group(group)[:,-6:], expected)
            term.noise = copy.deepcopy(noise)
            term.noise.func = lambda data, cfg: data + data.new_tensor([100.,-100.,100.,100.,-100.,100.])
            noisy_expected = (sample + sample.new_tensor([100.,-100.,100.,100.,-100.,100.])).clamp(min=term.clip[0],max=term.clip[1])
            torch.testing.assert_close(raw.observation_manager.compute_group(group)[:,-6:], noisy_expected)
        finally:
            term.func, term.noise = reader, noise
        assert sample[0,0] == 200.
    print('FINAL_WRENCH_CLIP_WITH_AND_WITHOUT_NOISE_OK', flush=True)
    robot, objects = raw.scene['robot'], raw.scene['object_collection']
    assert ('target_contact' in raw.reward_manager.active_terms) == (args.version != 3)
    assert raw.reward_manager.get_term_cfg('target_goal_distance').weight == (25. if args.version == 3 else 5.)
    assert raw.reward_manager.get_term_cfg('target_inverse_distance').weight == -12.97
    for name in ('target_goal_first_entry_bonus', 'target_goal_hold', 'target_goal_success_bonus'):
        assert (name in raw.reward_manager.active_terms) == (args.version != 3)
    assert ('target_goal_success' in raw.termination_manager.active_terms) == (args.version != 3)
    assert 'timeout' in raw.termination_manager.active_terms
    assert objects.num_objects == 17
    hand = robot.actuators['hand']
    torch.testing.assert_close(hand.effort_limit, torch.full_like(hand.effort_limit, 10.))
    torch.testing.assert_close(hand.velocity_limit, torch.ones_like(hand.velocity_limit))
    arm = raw.action_manager.get_term('arm_action')
    torch.testing.assert_close(robot.data.joint_vel_limits[:, arm._joint_ids],
                               torch.full((args.num_envs, 6), 3.14, device=raw.device))
    torch.testing.assert_close(robot.data.joint_effort_limits[:, arm._joint_ids],
                               torch.tensor([150.,150.,150.,28.,28.,28.], device=raw.device).expand(args.num_envs,-1))
    housing = robot.body_names.index('axia80_link')
    housing_mass = robot.root_physx_view.get_masses()[:, housing]
    torch.testing.assert_close(housing_mass, torch.full_like(housing_mass, .280))
    thresholds = {}
    for iteration, threshold in ((0,4.), (10000,4.), (10249,4.), (10250,3.95),
                                 (14500,3.1), (19750,2.05), (20000,2.), (90000,2.), (0,4.)):
        raw.set_learning_iteration(iteration)
        term = raw.termination_manager.get_term_cfg('arm_joint_velocity')
        assert term.params['threshold'] == threshold
        assert 'print_velocity' not in term.params
        thresholds[iteration] = threshold
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorRelativeV2EnvCfg
    v2 = WrenchRefactorRelativeV2EnvCfg()
    expected_v2 = WrenchRefactorRelativeEnvCfg().to_dict()
    expected_v2['observations'] = v2.observations.to_dict()
    assert v2.to_dict() == expected_v2

    for axis in range(3):
        command = torch.zeros(args.num_envs, 6, device=raw.device)
        command[:, axis] = 1.
        arm.process_actions(command)
        delta = arm._osc.desired_ee_pose_b[:, :3] - arm._ee_pose_b[:, :3]
        torch.testing.assert_close(delta, quat_apply(arm._ee_pose_b[:, 3:], command[:, :3] * .03), atol=1e-6, rtol=0.)
    arm.process_actions(torch.zeros(args.num_envs, 6, device=raw.device))
    mass = raw._clutter_physics.mass_buffer.clone()
    bounds = raw._revision_bounds_t.clone()
    materials = [m.properties.clone() for m in raw._clutter_physics.materials]
    ids = torch.arange(0, args.num_envs, 2, device=raw.device)
    other = torch.arange(1, args.num_envs, 2, device=raw.device)
    before = objects.data.object_link_state_w[other].clone()
    obs_before = raw._initial_obstacle_observation[other].clone()
    motion = raw.reward_manager.get_term_cfg('target_goal_distance').func
    inverse = raw.reward_manager.get_term_cfg('target_inverse_distance').func
    anchor_before = motion._anchor[other].clone()
    previous_before = inverse._previous[other].clone()
    raw.reset(env_ids=ids)
    target_xy = objects.data.object_pos_w[torch.arange(raw.num_envs, device=raw.device), raw.target_id[:, 0], :2]
    torch.testing.assert_close(motion._anchor[ids], target_xy[ids])
    torch.testing.assert_close(inverse._previous[ids], target_xy[ids])
    torch.testing.assert_close(motion._anchor[other], anchor_before)
    torch.testing.assert_close(inverse._previous[other], previous_before)
    torch.testing.assert_close(objects.data.object_link_state_w[other], before)
    torch.testing.assert_close(raw._initial_obstacle_observation[other], obs_before)
    actions = torch.zeros(args.num_envs, 8, device=raw.device)
    actions[:, -2:] = 1.
    max_parking_drift = 0.
    max_initial_bottom_error = 0.
    for iteration in (0, 9999, 10000, 13000, 25000):
        raw.set_learning_iteration(iteration)
        obs, _ = env.reset()
        active = raw._active_object_mask.clone()
        parked = ~active
        positions = objects.data.object_pos_w.clone()
        shelf = raw.scene['shelf'].data
        local = quat_apply_inverse(shelf.root_quat_w[:, None].expand(-1,17,-1), positions - shelf.root_pos_w[:,None])
        bottom = local[..., 2] + bounds[..., 0, 2]
        error = (bottom[active] - 1.05).abs().max().item()
        assert error < 1e-5, error
        max_initial_bottom_error = max(max_initial_bottom_error, error)
        command = raw.command_manager.get_term('target_goal_pos')
        distance = (command.command[:, :2] - command.target_init_state_w[:, :2]).norm(dim=-1)
        assert (distance >= .06 - 1e-6).all() and (distance <= .12 + 1e-6).all()
        assert (positions[..., 2][parked] > 40).all()
        # Follow only slots whose environments have not automatically reset.
        live = torch.ones(args.num_envs, dtype=torch.bool, device=raw.device)
        for step in range(36):
            obs, reward, _, _, _ = env.step(actions)
            live &= ~raw.reset_buf
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
            assert set(obs) == set(expected_dims)
            from inspire_sweep_baseline.mdp.observation_functions.wrench_refactor import initial_obstacle_positions_in_eef
            torch.testing.assert_close(obs['obstacles'], initial_obstacle_positions_in_eef(raw), rtol=0, atol=0)
            eef = raw.scene['ee_frame'].data
            root_data = robot.data
            groups = ('policy', 'critic') if args.version == 2 else ('policy',)
            for group in groups:
                limits = obs[group].new_tensor(cfg.wrist_wrench_observation_limits)
                assert (obs[group][:, 52:58].abs() <= limits).all()
                target_term = getattr(raw.observation_manager.cfg, group).target_obs_state.func
                if group == 'critic':
                    world = objects.data.object_pos_w[torch.arange(raw.num_envs, device=raw.device), raw.target_id[:, 0].long()]
                else:
                    world = raw.command_manager.get_term('target_goal_pos').target_init_state_w[:, :3]
                noisy_world = world + quat_apply(root_data.root_quat_w, target_term._bias_b)
                target_expected = quat_apply_inverse(eef.target_quat_w[:, 0], noisy_world - eef.target_pos_w[:, 0])
                torch.testing.assert_close(obs[group][:, 21:24], target_expected, atol=2e-5, rtol=1e-5)
            rows = obs['obstacles'].reshape(args.num_envs, 16, 4)
            assert not rows[rows[...,0] == 0].any()
            tactile = obs['policy'][:,35:52]
            assert ((tactile == -1) | (tactile == 0) | (tactile == 1) | (tactile == 2)).all()
            mask = parked & live[:, None]
            if mask.any():
                drift = (objects.data.object_pos_w - positions).norm(dim=-1)[mask].max().item()
                max_parking_drift = max(max_parking_drift, drift)
                assert drift < 1e-4, drift
        assert torch.equal(raw._clutter_physics.mass_buffer, mass)
        assert torch.equal(raw._revision_bounds_t, bounds)
        assert all(torch.equal(m.properties, old) for m,old in zip(raw._clutter_physics.materials,materials))
    if args.version == 3:
        # Manager-level success/timeout distinction; preserve physical failure terms.
        raw.reset()
        command = raw.command_manager.get_term('target_goal_pos')
        rows = torch.arange(raw.num_envs, device=raw.device)
        command.command[:, :2] = objects.data.object_pos_w[rows, raw.target_id[:, 0], :2]
        raw.episode_length_buf[:] = 150  # Former three-second success hold.
        raw.termination_manager.compute()
        assert not raw.termination_manager.get_term('timeout').any()
        raw.episode_length_buf[:] = raw.max_episode_length
        raw.termination_manager.compute()
        assert raw.termination_manager.get_term('timeout').all()
        assert raw.termination_manager.time_outs.all()
    report = dict(motion_history_partial_reset=True, inverse_weight=-12.97,
                  success_terms_removed=(args.version == 3), version=args.version, num_envs=args.num_envs, config_allowlist=True, parent_isolation=True,
                  partial_reset=True, rollout_steps=180, observation_dims=expected_dims,
                  max_initial_bottom_error_m=max_initial_bottom_error,
                  max_parking_drift_m=max_parking_drift, fixed_physics=True, eef_osc=True,
                  arm_velocity_limits=[3.14]*6, arm_effort_limits=[150.]*3+[28.]*3,
                  axia_mass_kg=float(housing_mass[0]), velocity_curriculum_thresholds=thresholds)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'probe_{args.num_envs}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
