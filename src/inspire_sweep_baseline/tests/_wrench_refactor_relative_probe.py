"""Real config allowlist and GPU spawn/reset/observation checks for Relative v0."""
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
    expected['rewards']['target_goal_first_entry_bonus']['weight'] = 10.
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
    print('CONFIG_ALLOWLIST_AND_PARENT_ISOLATION_OK', flush=True)

    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 42
    cfg.log_dir = str(args.output)
    env = gym.make('Inspire-Wrench-Refactor-Relative-v0', cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    assert {k:v.shape[-1] for k,v in obs.items()} == {'policy':58, 'obstacles':64}
    robot, objects = raw.scene['robot'], raw.scene['object_collection']
    assert objects.num_objects == 17
    hand = robot.actuators['hand']
    torch.testing.assert_close(hand.effort_limit, torch.full_like(hand.effort_limit, 10.))
    torch.testing.assert_close(hand.velocity_limit, torch.ones_like(hand.velocity_limit))
    arm = raw.action_manager.get_term('arm_action')
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
    raw.reset(env_ids=ids)
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
            assert set(obs) == {'policy', 'obstacles'}
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
    report = dict(num_envs=args.num_envs, config_allowlist=True, parent_isolation=True,
                  partial_reset=True, rollout_steps=180, observation_dims={'policy':58,'obstacles':64},
                  max_initial_bottom_error_m=max_initial_bottom_error,
                  max_parking_drift_m=max_parking_drift, fixed_physics=True, eef_osc=True)
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
