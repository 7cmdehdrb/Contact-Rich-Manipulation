"""Actual Revision-v2 CFG isolation, GPU reset/rollout and reward lifecycle smoke."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--test-variant', action='store_true')
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=32)
parser.add_argument('--output', type=Path, default=Path('outputs/revision_v2_validation'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.revision_envs import RevisionV1EnvCfg, RevisionV2EnvCfg, RevisionV2TestEnvCfg
    from inspire_sweep_baseline.agent.revision_ppo_cfg import (
        RevisionV1PPORunnerCfg, RevisionV2PPORunnerCfg, RevisionV2TestPPORunnerCfg,
    )

    def differences(left, right, prefix=''):
        if isinstance(left, dict) and isinstance(right, dict):
            result = []
            for key in sorted(left.keys() | right.keys()):
                path = f'{prefix}.{key}' if prefix else key
                if key not in left or key not in right:
                    result.append(path)
                else:
                    result += differences(left[key], right[key], path)
            return result
        return [] if left == right else [prefix]

    parent = RevisionV1EnvCfg()
    before = parent.to_dict()
    child, test = RevisionV2EnvCfg(), RevisionV2TestEnvCfg()
    assert type(child).__bases__ == (RevisionV1EnvCfg,)
    assert type(test).__bases__ == (RevisionV2EnvCfg,)
    test_diff = differences(child.to_dict(), test.to_dict())
    assert test_diff == ['scene.object_collection.rigid_objects.mass_cylinder.spawn.kind_override'], test_diff
    parent_diff = differences(before, child.to_dict())
    allowed = ('rewards.', 'contact_height_limit', 'goal_reward_height_limit')
    assert all(path.startswith(allowed) for path in parent_diff), parent_diff
    assert RevisionV1EnvCfg().to_dict() == before
    ppo = RevisionV1PPORunnerCfg().to_dict()
    for cls in (RevisionV2PPORunnerCfg, RevisionV2TestPPORunnerCfg):
        assert differences(ppo, cls().to_dict()) == ['experiment_name']
    assert child.events.object_spawn.params['gray_obstacles'] == 3
    assert child.events.object_spawn.params['white_obstacles'] == 3
    assert child.rewards.target_goal_hold.params == {'full_reward_radius': .025, 'outer_radius': .035}
    assert child.goal_hold_outer_radius is None
    assert child.contact_height_limit == child.goal_reward_height_limit == .095
    assert not getattr(test, 'debug_print_eef_height', False)
    assert not any(k.startswith('temp_eef_height') for k in vars(test.scene))
    assert child.rewards.target_goal_hold.weight == 10
    assert child.rewards.object_acceleration_penalty.params['free_acceleration'] == 15

    cfg = test if args.test_variant else child
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.log_dir = str(args.output)
    task = 'Isaac-Inspire-Revision-Test-v2' if args.test_variant else 'Isaac-Inspire-Revision-v2'
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    assert {k: v.shape[-1] for k,v in obs.items()} == {'policy':57,'obstacles':64,'privileged':3,'critic_obstacles':64}
    assert raw.scene['object_collection'].num_objects == 10
    assert len(raw.reward_manager.active_terms) == 15
    assert raw.curriculum_manager.active_terms == ['obstacle_population','target_free_speed','target_goal_direction_deadzone']
    kinds = np.asarray(raw._revision_object_kinds)
    scales = np.asarray(raw._final_object_scales)
    assert np.all((scales >= .9) & (scales <= 1.1))
    if args.test_variant:
        assert np.all(kinds[:,0] == 0)
    assert len(np.unique(kinds[:,1:])) == 8
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = 'test' if args.test_variant else 'v2'
    np.savez(args.output / f'spawn_{suffix}_{args.num_envs}.npz', kinds=kinds, scales=scales)
    if args.test_variant and (args.output / f'spawn_v2_{args.num_envs}.npz').exists():
        previous = np.load(args.output / f'spawn_v2_{args.num_envs}.npz')
        np.testing.assert_array_equal(previous['scales'], scales)
        np.testing.assert_array_equal(previous['kinds'][:,1:], kinds[:,1:])

    fixed = ('object_acceleration_penalty','obstacle_contact_penalty','wrist_wrench_change_penalty')
    def fixed_values():
        return {name: (raw.reward_manager.get_term_cfg(name).weight,
                       dict(raw.reward_manager.get_term_cfg(name).params)) for name in fixed}
    initial_fixed = fixed_values()
    counts = []
    for iteration, requested in [(0,(3,3)),(9999,(3,3)),(10000,(3,4)),(11999,(3,4)),
                                 (12000,(4,4)),(13999,(4,4)),(14000,(4,5)),(90000,(4,5)),(0,(3,3))]:
        mask = raw._active_object_mask.clone()
        raw.set_learning_iteration(iteration)
        torch.testing.assert_close(mask, raw._active_object_mask)
        params = raw.event_manager.get_term_cfg('object_spawn').params
        assert (params['gray_obstacles'],params['white_obstacles']) == requested
        assert fixed_values() == initial_fixed
        obs, _ = env.reset()
        counts.append({'iteration':iteration,'requested':requested,
                       'actual_mean':raw._final_actual_population.float().mean(0).tolist()})

    actions = torch.zeros((args.num_envs,raw.action_manager.total_action_dim),device=raw.device)
    actions[:,-2:] = 1
    for iteration in (0,14000):
        raw.set_learning_iteration(iteration)
        env.reset()
        for _ in range(args.steps):
            obs,reward,terminated,truncated,_ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
    ids = torch.arange(0,args.num_envs,2,device=raw.device)
    other = torch.arange(1,args.num_envs,2,device=raw.device)
    entry = raw.reward_manager.get_term_cfg('target_goal_first_entry_bonus').func
    entry._entered[:] = True
    before_positions = raw.scene['object_collection'].data.object_link_state_w[other].clone()
    raw.reset(env_ids=ids)
    assert not entry._entered[ids].any() and entry._entered[other].all()
    torch.testing.assert_close(raw.scene['object_collection'].data.object_link_state_w[other], before_positions)

    # Real env.step lifecycle: align selected goals after physics, before the
    # unchanged termination/reward managers. Row 2 is actually airborne/fast.
    env.reset()
    entry._entered[1] = True
    raw.episode_length_buf[:2] = raw.max_episode_length - 1
    objects = raw.scene['object_collection']
    state = objects.data.object_state_w.clone()
    state[2,0,2] = 3.0
    state[2,0,7] = 2.0
    objects.write_object_state_to_sim(state)
    original_termination = raw.termination_manager.compute
    original_reward = raw.reward_manager.compute
    captured = {}

    def termination_fixture():
        raw.command_manager.get_term('target_goal_pos').pos_command_w[:3,:2] = objects.data.object_pos_w[:3,0,:2]
        result = original_termination()
        captured['success'] = raw.termination_manager.get_term('target_goal_success_at_timeout')[:3].clone()
        captured['speed_failure'] = raw.termination_manager.get_term('target_speed')[:3].clone()
        captured['velocity'] = objects.data.object_lin_vel_w[:3,0].norm(dim=-1).clone()
        return result

    def reward_capture(dt):
        result = original_reward(dt)
        captured['terms'] = raw.reward_manager._step_reward[:3].clone() * dt
        captured['entered_before_reset'] = entry._entered[:3].clone()
        return result

    raw.termination_manager.compute = termination_fixture
    raw.reward_manager.compute = reward_capture
    try:
        _, _, terminated, truncated, _ = env.step(actions)
    finally:
        raw.termination_manager.compute = original_termination
        raw.reward_manager.compute = original_reward
    names = raw.reward_manager.active_terms
    entry_amount = captured['terms'][:,names.index('target_goal_first_entry_bonus')]
    timeout_amount = captured['terms'][:,names.index('target_goal_timeout_bonus')]
    expected_entry = 25 * torch.exp(-((captured['velocity'] - .01).clamp_min(0) / .02).square())
    expected_entry[1] = 0
    torch.testing.assert_close(entry_amount, expected_entry)
    torch.testing.assert_close(timeout_amount, torch.tensor([75.,75.,0.],device=raw.device))
    assert captured['speed_failure'][2] and captured['velocity'][2] > 1
    assert captured['entered_before_reset'].all() and not entry._entered[:3].any()
    assert (terminated[:3] | truncated[:3]).all()
    event_lifecycle = dict(entry=entry_amount.tolist(), timeout=timeout_amount.tolist(),
                           high_speed_failure=bool(captured['speed_failure'][2]),
                           paid_before_automatic_reset=True)

    report = dict(task=task,num_envs=args.num_envs,steps=2*args.steps,partial_reset=True,
                  event_lifecycle=event_lifecycle,parent_diff=parent_diff,test_diff=test_diff,rewards=raw.reward_manager.active_terms,
                  fixed_values=initial_fixed,counts=counts,
                  target_kinds=np.bincount(kinds[:,0],minlength=8).tolist(),
                  obstacle_kinds=np.bincount(kinds[:,1:].reshape(-1),minlength=8).tolist(),
                  scale_range=[float(scales.min()),float(scales.max())])
    (args.output / f'probe_{suffix}_{args.num_envs}.json').write_text(json.dumps(report,indent=2))
    print('REVISION_V2_PROBE ' + json.dumps(report),flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
