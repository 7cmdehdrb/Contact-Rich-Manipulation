"""Real Sim import/reset/step checks for every registered config and Final v1/v2.

Run with Isaac Lab Python; reports appear only after successful assertions.
"""
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
parser.add_argument('--assets', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs.final_envs import FinalEnvCfg, FinalV1EnvCfg, FinalV2EnvCfg
    from pxr import UsdPhysics
    import isaaclab.sim as sim

    report = {'configs': [], 'smoke': []}
    for task in gym.registry:
        if task.startswith('Isaac-Inspire-'):
            cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
            agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
            assert cfg and agent
            report['configs'].append(task)
    assert FinalV1EnvCfg.__bases__ == (FinalEnvCfg,)
    assert FinalV2EnvCfg.__bases__ == (FinalV1EnvCfg,)
    if args.assets:
        import numpy as np
        import omni.usd
        from types import SimpleNamespace as NS
        from pxr import Usd, UsdGeom
        from inspire_sweep_baseline.assets.final_spawners import FinalShelfObjectCfg, spawn_shelf_object
        from inspire_sweep_baseline.assets.scenes.final import _target_spawn
        from inspire_sweep_baseline.mdp.event_functions.final import randomize_object_sizes
        omni.usd.get_context().new_stage()
        stage = sim.get_current_stage()
        template = _target_spawn()
        for row, kind in enumerate(('cube', 'mug', 'bottle', 'cylinder')):
            path = f'/World/envs/env_{row}/mass_cylinder'
            cfg = (template if kind == 'cylinder' else FinalShelfObjectCfg(kind=kind,
                   rigid_props=template.rigid_props, mass_props=template.mass_props,
                   collision_props=template.collision_props))
            cfg.func(path, cfg, translation=(-.7, row*2.5, 1.05))
        env = NS(num_envs=4, sim=NS(is_playing=lambda:False), cfg=NS(seed=42,
            target_bounding_box=True, scene=NS(object_collection=NS(rigid_objects={'mass_cylinder':None}))))
        collisions = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        cylinder_mass = UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/envs/env_3/mass_cylinder'))
        original_com = cylinder_mass.GetCenterOfMassAttr().Get()
        original_inertia = np.array(cylinder_mass.GetDiagonalInertiaAttr().Get())
        randomize_object_sizes(env, None)
        assert cylinder_mass.GetCenterOfMassAttr().Get() == original_com
        np.testing.assert_allclose(cylinder_mass.GetDiagonalInertiaAttr().Get(),
                                   original_inertia * env._final_object_scales[3, 0] ** 2, rtol=1e-6)
        assert np.isfinite(env._final_object_bounds).all()
        assert np.allclose(env._final_object_bounds[:, 0, 0, 2], 0., atol=1e-7)
        assert .9 <= env._final_object_scales.min() <= env._final_object_scales.max() <= 1.1
        assert collisions == [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        for row in range(4):
            root = stage.GetPrimAtPath(f'/World/envs/env_{row}/mass_cylinder')
            marker = stage.GetPrimAtPath(str(root.GetPath()) + '/TargetBoundingBox')
            assert marker.IsA(UsdGeom.BasisCurves) and not marker.HasAPI(UsdPhysics.CollisionAPI)
            assert len(UsdGeom.BasisCurves(marker).GetPointsAttr().Get()) == 24
            # Exclude the render-only marker before checking object support.
            stage.RemovePrim(str(marker.GetPath()))
            box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render']).ComputeWorldBound(root).ComputeAlignedRange()
            assert abs(box.GetMin()[2]-1.05) < 1e-6
            assert box.GetMax()[2] > 1.10
        report['assets'] = 'cube/mug/bottle/cylinder: USD spawn, size, bottom bounds, render-only box passed'
    for version in (1, 2):
        task = f'Isaac-Inspire-Final-v{version}'
        cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
        agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
        assert len(cfg.scene.object_collection.rigid_objects) == 21
        assert agent.actor.obstacle_observation_slots == agent.critic.obstacle_observation_slots == 20
        assert cfg.terminations.success.time_out and cfg.terminations.time_out.time_out
        if not args.smoke:
            continue
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device, 42
        cfg.target_bounding_box = version == 2
        env = gym.make(task, cfg=cfg)
        raw = env.unwrapped
        obs, _ = env.reset(seed=42)
        assert {key: value.shape[-1] for key, value in obs.items()} == {'policy':57, 'obstacles':80, 'privileged':3}
        assert .9 <= raw._final_object_scales.min() <= raw._final_object_scales.max() <= 1.1
        for row in range(4):
            marker = sim.get_current_stage().GetPrimAtPath(f'/World/envs/env_{row}/mass_cylinder/TargetBoundingBox')
            assert bool(marker) == (version == 2)
            if marker:
                assert not marker.HasAPI(UsdPhysics.CollisionAPI)
        for iteration in (0, 3000, 10000, 15000):
            raw.set_learning_iteration(iteration)
            env.reset()
            assert (raw._final_actual_population <= 8).all()
            for _ in range(10):
                obs, reward, _, _, _ = env.step(torch.zeros((4,8),device=raw.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(x).all() for x in obs.values())
        raw.episode_length_buf[:] = raw.max_episode_length
        done = raw.termination_manager
        done.compute()
        assert (done.get_term('success') | done.get_term('time_out')).all()
        assert not (done.get_term('success') & done.get_term('time_out')).any()
        report['smoke'].append(task)
        env.close()
    out = Path('outputs/final_versions_validation')
    out.mkdir(parents=True, exist_ok=True)
    (out / ('smoke.json' if args.smoke else 'config.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
finally:
    app.close()
