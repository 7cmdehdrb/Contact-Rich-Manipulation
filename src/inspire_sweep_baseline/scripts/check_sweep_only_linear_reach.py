"""Check the direct child contract and GPU reset/step/partial-reset Reach path."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--steps', type=int, default=8)
parser.add_argument('--output', type=Path, default=Path('outputs/sweep_only_linear_reach/smoke.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.steps < 1:
    parser.error('Need at least two environments and one step')
app = AppLauncher(args).app


def validate():
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from inspire_sweep_baseline.assets.hand_kinematics import HAND_DRIVE_UPPER_LIMITS
    from inspire_sweep_baseline.mdp.common.stabilize_sweep_contact import centroid
    from inspire_sweep_baseline.mdp.reward_functions.stabilize_sweep import linear_reach

    parent_id = 'Inspire-Sweep-Only-v2-Test-ActionPenalty'
    child_id = parent_id + '-LinearReach'
    parent = load_cfg_from_registry(parent_id, 'env_cfg_entry_point')
    before = parent.to_dict()
    child = load_cfg_from_registry(child_id, 'env_cfg_entry_point')
    assert type(child).__bases__ == (type(parent),)
    after = child.to_dict()
    reach = after['rewards']['reaching']
    assert child.rewards.reaching.func is linear_reach
    assert reach['weight'] == .8
    assert reach['params'] == {'near_radius': .1, 'decay_distance': .35, 'frame_name': 'ee_frame'}
    after['rewards']['reaching'] = before['rewards']['reaching']
    assert after == before, 'Unexpected environment change outside Reach'
    assert parent.to_dict() == before
    assert load_cfg_from_registry(parent_id, 'env_cfg_entry_point').to_dict() == before

    parent_runner = load_cfg_from_registry(parent_id, 'rsl_rl_cfg_entry_point')
    runner = load_cfg_from_registry(child_id, 'rsl_rl_cfg_entry_point')
    assert type(runner).__bases__ == (type(parent_runner),)
    rp, rc = parent_runner.to_dict(), runner.to_dict()
    assert rc.pop('experiment_name') == 'inspire_sweep_only_v2_test_action_penalty_linear_reach'
    rp.pop('experiment_name')
    assert rc == rp and runner.clip_actions is None
    assert gym.spec(parent_id).entry_point == gym.spec(child_id).entry_point

    child.scene.num_envs = args.num_envs
    child.seed = 42
    child.debug_surface_points = False
    child.scene.ee_frame.debug_vis = False
    child.commands.target_goal_pos.debug_vis = False
    env = gym.make(child_id, cfg=child)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner.clip_actions)
    base = env.unwrapped
    result = {'parent': parent_id, 'task': child_id, 'seed': 42, 'num_envs': args.num_envs,
              'configuration_difference': 'Reach function, weight and distance parameters only',
              'ppo_difference': 'experiment_name only', 'steps': args.steps}
    try:
        assert str(base.device).startswith('cuda'), base.device
        wrapped.reset()
        raw_center = centroid(base)
        frame = base.scene['ee_frame'].data.target_pos_w[:, 0]
        result['initial_center_distance_m'] = (frame - raw_center).norm(dim=-1).tolist()
        root = base.scene['object_collection'].data.object_link_pos_w[:, 0]
        result['initial_root_distance_m'] = (frame - root).norm(dim=-1).tolist()

        # Use the actual spawned/rotated object centers, with only a virtual
        # EEF for exact curve probes; no physics buffers are overwritten.
        distances = torch.tensor([.05, .10, .15, .20, .30, .35, .45, .60], device=base.device)
        eef = raw_center[:1].expand(len(distances), -1).clone()
        eef[:, 0] += distances
        data = base.scene['object_collection'].data
        objects = NS(data=NS(object_link_pos_w=data.object_link_pos_w[:1].expand(len(distances), -1, -1),
                             object_link_quat_w=data.object_link_quat_w[:1].expand(len(distances), -1, -1)))
        probe = NS(scene={'object_collection': objects, 'ee_frame': NS(data=NS(target_pos_w=eef[:, None]))},
                   _revision_bounds_t=base._revision_bounds_t[:1].expand(len(distances), -1, -1, -1))
        actual = linear_reach(probe)
        expected = torch.tensor([1., 1., 6/7, 5/7, 3/7, 2/7, 0., 0.], device=base.device)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        result['gpu_distance_probes_m'] = distances.tolist()
        result['gpu_raw_reach_probes'] = actual.tolist()

        action = torch.zeros((args.num_envs, 12), device=base.device)
        action[:, 6:] = action.new_tensor(HAND_DRIVE_UPPER_LIMITS) * .1
        index = base.reward_manager.active_terms.index('reaching')
        reach_samples = []
        for _ in range(args.steps):
            obs, reward, done, _ = wrapped.step(action)
            assert not done.any(), 'Unexpected early termination during short zero-arm rollout'
            assert obs['policy'].shape == (args.num_envs, 66)
            assert obs['obstacles'].shape == (args.num_envs, 64)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(obs[k]).all() for k in ('policy', 'obstacles'))
            current = linear_reach(base)
            torch.testing.assert_close(base.reward_manager._step_reward[:, index], .8 * current)
            # The shared manager still applies policy dt exactly once.
            torch.testing.assert_close(reward, base.reward_manager._step_reward.sum(-1) * base.step_dt,
                                       atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(base.action_manager.action, action)
            reach_samples.append((.8 * current).tolist())

        old_other = base.scene['object_collection'].data.object_link_state_w[1].clone()
        old_action = base.action_manager.action[1].clone()
        base.reset(env_ids=torch.tensor([0], device=base.device))
        torch.testing.assert_close(base.scene['object_collection'].data.object_link_state_w[1], old_other)
        torch.testing.assert_close(base.action_manager.action[1], old_action)
        torch.testing.assert_close(base.action_manager.action[0], torch.zeros_like(old_action))
        assert torch.isfinite(linear_reach(base)).all()
        obs, reward, _, _ = wrapped.step(action)
        assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
        result.update(device=str(base.device), weighted_reach_samples=reach_samples,
                      partial_reset='passed', post_reset_step='passed',
                      reward_manager_timestep='applied once', status='passed')
    finally:
        wrapped.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


try:
    validate()
except BaseException:
    # SimulationApp.close may exit immediately; print failures before cleanup.
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    app.close()
