"""Compare actual Final/ObjectRandom configs and optionally exercise Final in Sim."""

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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app


def compare(left, right, path='cfg'):
    """Check values, callable identity and dict order (manager execution order)."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and list(left) == list(right), (path, list(left), list(right))
        for key in left:
            compare(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f'{path}[{index}]')
    else:
        assert left == right, (path, left, right)


try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.runtime.final import FinalEnv

    old_task = 'Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v3'
    task = 'Isaac-Inspire-Final-v0'
    old = load_cfg_from_registry(old_task, 'env_cfg_entry_point')
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    old_agent = load_cfg_from_registry(old_task, 'rsl_rl_cfg_entry_point')
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    assert type(cfg).__bases__ == (ManagerBasedRLEnvCfg,)
    assert type(cfg.scene).__bases__ == (InteractiveSceneCfg,)
    assert FinalEnv.__bases__ == (ManagerBasedRLEnv,)
    for field in ('actions', 'commands', 'observations', 'rewards', 'terminations', 'events', 'curriculum'):
        assert type(getattr(cfg, field)).__bases__ == (object,), field
    compare(old.to_dict(), cfg.to_dict())
    assert agent.experiment_name == 'inspire_final_v0'
    new_ppo = agent.to_dict()
    new_ppo['experiment_name'] = old_agent.experiment_name
    compare(old_agent.to_dict(), new_ppo, 'ppo')
    # Same post-Hydra scene reconstruction, including nondefault capacities.
    for gray, white in ((8, 8), (3, 4)):
        old.scene.configure_population_capacity(gray, white, 16)
        cfg.scene.configure_population_capacity(gray, white, 16)
        compare(old.scene.to_dict(), cfg.scene.to_dict(), 'rebuilt_scene')
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    cfg.rewards.goal_hold.params['position_tolerance'] = 99
    assert old.rewards.goal_hold.params['position_tolerance'] == .02
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    report = {'config_comparison': 'passed', 'ordered_fields_and_callables': 'passed',
              'independent_cfg_and_runtime_bases': 'passed', 'scene_rebuild': 'passed',
              'ppo_comparison': 'passed except experiment_name'}
    out = Path('outputs/final_v0_validation')
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = 2, args.device, 42
        cfg.log_dir = str(out)
        env = gym.make(task, cfg=cfg)
        raw = env.unwrapped
        obs, _ = env.reset(seed=42)
        assert {k: v.shape[-1] for k, v in obs.items()} == {'policy': 55, 'obstacles': 64, 'privileged': 3}
        assert 'success' not in raw.termination_manager.active_terms
        for _ in range(10):
            obs, rewards, _, _, _ = env.step(torch.zeros((2, 8), device=raw.device))
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
        raw.episode_length_buf[0] = raw.max_episode_length - 1
        _, _, _, truncated, _ = env.step(torch.zeros((2, 8), device=raw.device))
        assert truncated[0].item()
        report['smoke'] = '2 envs, seed42, reset + 10 steps + forced timeout passed'
        env.close()
    (out / ('smoke.json' if args.smoke else 'config.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
finally:
    app.close()
