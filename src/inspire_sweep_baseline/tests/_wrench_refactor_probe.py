"""Real Isaac Sim reset/rollout check, writing only outputs/wrench_refactor_validation."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--config-only', action='store_true')
parser.add_argument('--output', type=Path, default=Path('outputs/wrench_refactor_validation'))
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
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorEnvCfg
    cfg = WrenchRefactorEnvCfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 42
    cfg.log_dir = str(args.output)
    if args.config_only:
        from inspire_sweep_baseline.agent.wrench_refactor_ppo_cfg import WrenchRefactorPPORunnerCfg
        runner = WrenchRefactorPPORunnerCfg()
        def symbolic(value):
            if callable(value):
                return f'{value.__module__}:{value.__qualname__}'
            return str(value)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'resolved_config.json').write_text(json.dumps(
            {'env':cfg.to_dict(), 'runner':runner.to_dict()}, default=symbolic, indent=2))
        print(json.dumps({'config_import':True,'objects':len(cfg.scene.object_collection.rigid_objects),
                          'runner_groups':runner.obs_groups,'goal_radius':cfg.commands.target_goal_pos.goal_radius}),flush=True)
        # No simulation has been created. Kit's no-device shutdown itself can
        # segfault on a driverless host; the config-only subprocess ends here.
        import os
        os._exit(0)
    started = time.perf_counter()
    env = gym.make('Inspire-Wrench-Refactor-v0', cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    assert {k: v.shape[-1] for k, v in obs.items()} == {'policy':74, 'critic':74, 'obstacles':64, 'critic_obstacles':64}
    assert raw.scene['object_collection'].num_objects == 17
    shelf_mass = raw.scene['shelf'].root_physx_view.get_masses()
    torch.testing.assert_close(shelf_mass, torch.full_like(shelf_mass, 30.))
    from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES
    robot = raw.scene['robot']
    hand_ids = [robot.body_names.index(name) for name in HAND_CONTACT_BODY_NAMES]
    hand_mass = robot.root_physx_view.get_masses()[:, hand_ids].sum(-1)
    torch.testing.assert_close(hand_mass, torch.full_like(hand_mass, .790))
    expected = np.array([0, *[i for i in range(8) for _ in range(2)]])
    assert np.array_equal(raw._revision_object_kinds, np.tile(expected, (args.num_envs,1)))
    mass = raw._clutter_physics.mass_buffer.clone()
    materials = [m.properties.clone() for m in raw._clutter_physics.materials]
    bounds = raw._revision_bounds_t.clone()
    ids = torch.arange(0, args.num_envs, 2, device=raw.device)
    other = torch.arange(1, args.num_envs, 2, device=raw.device)
    objects = raw.scene['object_collection']
    before = objects.data.object_link_state_w[other].clone()
    raw.reset(env_ids=ids)
    torch.testing.assert_close(objects.data.object_link_state_w[other], before)
    actions = torch.zeros(args.num_envs, 8, device=raw.device); actions[:, -2:] = 1.
    timings = []
    for iteration in (0,9999,10000,13000,16000,19000,22000,25000):
        raw.set_learning_iteration(iteration)
        t = time.perf_counter(); env.reset()
        if raw.device != 'cpu': torch.cuda.synchronize()
        timings.append(time.perf_counter()-t)
        for _ in range(36):
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
        assert torch.equal(raw._clutter_physics.mass_buffer, mass)
        assert torch.equal(raw._revision_bounds_t, bounds)
        assert all(torch.equal(m.properties, old) for m, old in zip(raw._clutter_physics.materials, materials))
    report = {'task':'Inspire-Wrench-Refactor-v0', 'num_envs':args.num_envs, 'elapsed_s':time.perf_counter()-started,
              'reset_seconds':timings, 'fixed_physics':True, 'partial_reset':True, 'finite_rollout':True}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output / f'probe_{args.num_envs}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
except Exception:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None: env.close()
    app.close()
