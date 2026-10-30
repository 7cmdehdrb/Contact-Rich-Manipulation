"""Compare actual v0_3/parent configs; optionally exercise reset/step on a GPU."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--output-dir', type=Path, default=Path('outputs/final_accuracy_v0_3_validation'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app


def compare(left, right, path='cfg'):
    """Compare every value and Manager term order, including callable paths."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and list(left) == list(right), path
        for key in left:
            compare(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f'{path}[{index}]')
    else:
        assert left == right, (path, left, right)


try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.mdp.reward_functions.goal_rewards import success_event

    prefix = 'Isaac-Inspire-Final-GoalAccuracy-'
    task = prefix + 'v0_3'
    parents = {version: load_cfg_from_registry(prefix + version, 'env_cfg_entry_point')
               for version in ('v0', 'v0_1', 'v0_2', 'v1')}
    snapshots = {version: cfg.to_dict() for version, cfg in parents.items()}
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    parent = parents['v0_2']
    assert type(cfg).__bases__ == (type(parent),)
    assert gym.spec(task).entry_point == gym.spec(prefix + 'v0_2').entry_point
    expected = parent.to_dict()
    expected['rewards']['target_contact']['weight'] = .3625
    expected['rewards']['target_goal_hold']['params']['position_tolerance'] = .03
    expected['rewards']['target_goal_distance'] = parents['v0'].rewards.goal_distance.to_dict()
    assert cfg.rewards.target_goal_timeout_bonus.func is success_event
    assert cfg.rewards.target_goal_timeout_bonus.weight == 100.
    assert cfg.rewards.target_goal_timeout_bonus.params == {'termination_name': 'target_goal_success_at_timeout'}
    expected['rewards']['target_goal_timeout_bonus'] = cfg.rewards.target_goal_timeout_bonus.to_dict()
    compare(expected, cfg.to_dict())
    ppo = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    parent_ppo = load_cfg_from_registry(prefix + 'v0_2', 'rsl_rl_cfg_entry_point')
    assert type(ppo).__bases__ == (type(parent_ppo),)
    compare({**parent_ppo.to_dict(), 'experiment_name': 'inspire_final_goal_accuracy_v0_3'}, ppo.to_dict(), 'ppo')
    for gray, white in ((10, 10), (0, 0), (3, 4)):
        left, right = parent.scene.copy(), cfg.scene.copy()
        for scene in (left, right):
            scene.configure_population_capacity(gray, white, 20)
        compare(left.to_dict(), right.to_dict(), 'rebuilt_scene')
    # Child changes must not leak into existing or newly created parents.
    isolated = cfg.copy()
    isolated.rewards.target_goal_hold.params['position_tolerance'] = .9
    for version, snapshot in snapshots.items():
        compare(snapshot, parents[version].to_dict(), version)
        compare(snapshot, load_cfg_from_registry(prefix + version, 'env_cfg_entry_point').to_dict(), version)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'task': task, 'config_delta': 'only four specified rewards; all ordered fields compared',
        'ppo_delta': 'experiment_name only', 'parent_isolation': list(parents),
        'scene_rebuild_capacities': [[10, 10], [0, 0], [3, 4]],
        'changed_rewards': {name: getattr(cfg.rewards, name).to_dict() for name in (
            'target_contact', 'target_goal_distance', 'target_goal_hold', 'target_goal_timeout_bonus')},
        'smoke_requested': args.smoke,
    }
    (args.output_dir/'config_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if args.smoke:
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = 2, args.device, 42
        cfg.log_dir = str(args.output_dir)
        env = gym.make(task, cfg=cfg)
        try:
            raw = env.unwrapped
            obs, _ = env.reset(seed=42)
            assert {key: value.shape[-1] for key, value in obs.items()} == {
                'policy': 57, 'obstacles': 80, 'privileged': 3}
            for iteration in (0, 6000, 12000, 0):
                raw.set_learning_iteration(iteration)
                env.reset()
                assert raw.reward_manager.get_term_cfg('target_goal_hold').params['position_tolerance'] == .03
                for _ in range(5):
                    _, reward, _, _, _ = env.step(torch.zeros(2, 8, device=raw.device))
                    assert torch.isfinite(reward).all()
            raw.episode_length_buf[:] = raw.max_episode_length
            raw.termination_manager.compute()
            success = raw.termination_manager.get_term('target_goal_success_at_timeout')
            miss = raw.termination_manager.get_term('target_goal_miss_at_timeout')
            assert (success | miss).all() and not (success & miss).any()
            bonus = raw.reward_manager.get_term_cfg('target_goal_timeout_bonus')
            torch.testing.assert_close(
                bonus.func(raw, **bonus.params) * bonus.weight * raw.step_dt, success.float() * 100.)
            (args.output_dir/'smoke.json').write_text(json.dumps({'reset_step_curriculum_timeout': 'passed'}) + '\n')
        finally:
            env.close()
finally:
    app.close()
