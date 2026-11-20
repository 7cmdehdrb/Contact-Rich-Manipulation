"""Compare the full parent contract and exercise observation/reward isolation on GPU."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--single-contact', action='store_true', help='Validate the binary-contact child instead of Sweep v0')
parser.add_argument('--output', type=Path, default=Path('outputs/sweep_v0/smoke.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def validate():
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from inspire_sweep_baseline.assets.hand_kinematics import HAND_DRIVE_UPPER_LIMITS
    from inspire_sweep_baseline.mdp.reward_functions.sweep_contact_preference import ContactSideSweepProgress
    from inspire_sweep_baseline.mdp.reward_functions.stabilize_sweep import sweep_progress

    parent_id = 'Inspire-Sweep-Only-v2-Test-ActionPenalty-LinearReach'
    task = 'Inspire-Sweep-Single-Contact-v0' if args.single_contact else 'Inspire-Sweep-v0'
    if args.single_contact and args.output == Path('outputs/sweep_v0/smoke.json'):
        args.output = Path('outputs/sweep_single_contact/smoke.json')
    parent = load_cfg_from_registry(parent_id, 'env_cfg_entry_point')
    before = parent.to_dict()
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    if args.single_contact:
        sweep_cfg = load_cfg_from_registry('Inspire-Sweep-v0', 'env_cfg_entry_point')
        assert type(cfg).__bases__ == (type(sweep_cfg),)
        assert cfg.to_dict() == sweep_cfg.to_dict()
    else:
        assert type(cfg).__bases__ == (ManagerBasedRLEnvCfg,)
    after = cfg.to_dict()
    assert cfg.rewards.sweep_progress.func is ContactSideSweepProgress
    assert cfg.rewards.sweep_progress.params == {'preferred_multiplier': 1.5, 'other_multiplier': .7}
    assert cfg.rewards.sweep_progress.weight == 50.
    assert cfg.events.object_spawn.params['excluded_x_half_angle'] == 0.
    assert cfg.rewards.obstacle_contact.params['free_force'] == 50.
    assert cfg.rewards.obstacle_contact.weight == -.04
    assert cfg.rewards.wrench_change.params['free_force_change'] == 30.
    assert cfg.rewards.wrench_change.params['free_moment_change'] == 3.
    assert cfg.rewards.wrench_change.weight == -.1
    expected_scale = (1 / 150.,) * 3 + (1 / 8.,) * 3
    assert cfg.observations.policy.wrist_wrench.scale == expected_scale
    # Restore exactly the requested differences in the comparison copy only.
    after['rewards']['sweep_progress'] = before['rewards']['sweep_progress']
    for name in ('obstacle_contact', 'wrench_change'):
        after['rewards'][name] = before['rewards'][name]
    assert cfg.curriculum == {}
    assert {name for name, term in before['curriculum'].items() if term is not None} == {
        'obstacle_free_force', 'obstacle_weight', 'wrench_free_force_change',
        'wrench_free_moment_change', 'wrench_weight'}
    after['curriculum'] = before['curriculum']
    after['observations']['policy']['wrist_wrench']['scale'] = before['observations']['policy']['wrist_wrench']['scale']
    assert after == before, 'Unrequested environment difference'
    assert parent.to_dict() == before
    assert load_cfg_from_registry(parent_id, 'env_cfg_entry_point').to_dict() == before
    parent_runner = load_cfg_from_registry(parent_id, 'rsl_rl_cfg_entry_point')
    runner = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    rp, rc = parent_runner.to_dict(), runner.to_dict()
    assert rc.pop('experiment_name') == ('inspire_sweep_single_contact_v0' if args.single_contact else 'inspire_sweep_v0')
    rp.pop('experiment_name')
    assert rc == rp
    if args.single_contact:
        sweep_runner = load_cfg_from_registry('Inspire-Sweep-v0', 'rsl_rl_cfg_entry_point')
        assert type(runner).__bases__ == (type(sweep_runner),)
    else:
        assert type(runner).__bases__ == (RslRlOnPolicyRunnerCfg,)
    runtime_name = 'SweepSingleContactEnv' if args.single_contact else 'SweepEnv'
    assert gym.spec(task).entry_point == f'inspire_sweep_baseline.runtime.sweep_v0:{runtime_name}'
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.debug_surface_points = cfg.scene.ee_frame.debug_vis = cfg.commands.target_goal_pos.debug_vis = False
    wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=runner.clip_actions)
    base = wrapped.unwrapped
    result = {'task': task, 'parent': parent_id, 'configuration': 'requested differences only',
              'ppo': 'experiment name only', 'num_envs': 4, 'seed': 42}
    try:
        wrapped.reset()
        if args.single_contact:
            from inspire_sweep_baseline.mdp.common.sweep_v0_state import SingleContactSweepState
            assert type(base._stabilize_state) is SingleContactSweepState
            probe = SingleContactSweepState(4, base.device)
            probe.contact_history[:, :2] = torch.tensor([[False, False], [True, False],
                                                        [True, True], [True, True]], device=base.device)
            probe.previous_distance.fill_(1.)
            probe.finish(torch.tensor([[.01, 0., 0.]], device=base.device).expand(4, -1),
                         torch.tensor([[1., 0., 0.]], device=base.device).expand(4, -1),
                         torch.tensor([[2., 0., 0.]], device=base.device).expand(4, -1),
                         torch.tensor([True, True, True, False], device=base.device),
                         torch.ones(4, device=base.device, dtype=torch.bool),
                         torch.full((4,), 1.1, device=base.device), torch.ones(4, device=base.device))
            assert not hasattr(probe, 'contact_count')
            assert probe.contact_recognized.dtype == torch.bool
            assert probe.contact_recognized.tolist() == [False, True, True, True]
            torch.testing.assert_close(probe.positive, torch.tensor([0., .01, .01, 0.], device=base.device))
            result['binary_contact_gpu_probe'] = 'boolean recognition, no contact_count; 0/1 factor; current-force gate preserved'
        assert str(base.device).startswith('cuda')
        assert not base.curriculum_manager.active_terms
        side_cfg = base.reward_manager.get_term_cfg('sweep_progress')
        side = side_cfg.func
        assert isinstance(side, ContactSideSweepProgress)
        # Test the actual observation manager, including extreme physical values,
        # without writing to physical sensor buffers or normalizing their source.
        term = base.observation_manager.cfg.policy.wrist_wrench
        old_func, old_noise = term.func, term.noise
        sample = torch.tensor([[0., 75., -150., 4., -8., 0.],
                               [300., -300., 150., 16., -16., 8.],
                               [6.8596, 37.6125, 17.0149, -4.734, .806, -.226],
                               [0., 0., 0., 0., 0., 0.]], device=base.device)
        limits = sample.new_tensor(base.cfg.wrist_wrench_observation_limits)
        term.func, term.noise = lambda env: sample.clone(), None
        original_sample = sample.clone()
        wrench = base.reward_manager.get_term_cfg('wrench_change').func
        params = base.reward_manager.get_term_cfg('wrench_change').params
        old_key = wrench._cached_key
        base.sensor_data_fresh.fill_(True)
        try:
            actual = base.observation_manager.compute_group('policy')[:, -6:]
            expected = sample.clamp(-limits, limits) / limits
            torch.testing.assert_close(actual, expected)
            assert actual.abs().max() <= 1.
            torch.testing.assert_close(sample, original_sample)
            # The real penalty directly reads the physical observation callable;
            # manager clip/scale/noise are not part of this path.
            wrench.reset()
            wrench._cached_key = None
            wrench(base, **params)
            torch.testing.assert_close(wrench._previous_wrench, original_sample)
            result['normalized_probe'] = actual.tolist()
            result['physical_wrench_penalty_history'] = 'unchanged N/Nm, including values beyond observation clip'
        finally:
            term.func, term.noise = old_func, old_noise
            wrench.reset()
            wrench._cached_key = old_key
        action = torch.zeros(4, 12, device=base.device)
        action[:, 6:] = action.new_tensor(HAND_DRIVE_UPPER_LIMITS) * .1
        index = base.reward_manager.active_terms.index('sweep_progress')
        for _ in range(8):
            obs, reward, done, _ = wrapped.step(action)
            assert not done.any()
            assert obs['policy'].shape == (4, 66) and obs['obstacles'].shape == (4, 64)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert obs['policy'][:, -6:].abs().max() <= 1.
            if args.single_contact:
                assert not hasattr(base._stabilize_state, 'contact_count')
                assert base._stabilize_state.contact_recognized.dtype == torch.bool
            torch.testing.assert_close(base.reward_manager._step_reward[:, index], 50 * side(base, **side_cfg.params))
            torch.testing.assert_close(reward, base.reward_manager._step_reward.sum(-1) * base.step_dt)
        # Use the live reward instance and real physics side evidence; classified
        # synthetic cases are tested separately on CPU, not claimed as contacts.
        palm, dorsal = side._surface_loads(base)
        assert torch.isfinite(palm).all() and torch.isfinite(dorsal).all()
        assert ((palm >= 0) & (dorsal >= 0)).all()
        assert torch.isfinite(sweep_progress(base)).all()
        other = base.scene['object_collection'].data.object_link_state_w[1].clone()
        base.reset(env_ids=torch.tensor([0], device=base.device))
        torch.testing.assert_close(base.scene['object_collection'].data.object_link_state_w[1], other)
        obs, reward, _, _ = wrapped.step(action)
        assert obs['policy'][:, -6:].abs().max() <= 1. and torch.isfinite(reward).all()
        result.update(device=str(base.device), steps=8, partial_reset='passed', status='passed')
    finally:
        wrapped.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


try:
    validate()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    app.close()
