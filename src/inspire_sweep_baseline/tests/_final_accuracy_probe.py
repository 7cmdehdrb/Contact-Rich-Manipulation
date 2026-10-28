"""Actual Isaac imports, complete config delta, USD cuboid bounds; optional reset/step."""
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
try:
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.final_envs import (
        FinalGoalAccuracyEnvCfg, FinalGoalAccuracyV1EnvCfg, FinalGoalAccuracyV0_1EnvCfg)
    from inspire_sweep_baseline.assets.final_spawners import FinalShelfObjectCfg
    from inspire_sweep_baseline.assets.scenes.final import _target_spawn
    from inspire_sweep_baseline.mdp.event_functions.final import randomize_object_sizes
    from isaaclab_tasks.utils import load_cfg_from_registry
    from types import SimpleNamespace as NS

    output = Path('outputs/final_accuracy_v0_1_validation')
    output.mkdir(parents=True, exist_ok=True)
    task = 'Isaac-Inspire-Final-GoalAccuracy-v0_1'
    parent, sibling = FinalGoalAccuracyEnvCfg(), FinalGoalAccuracyV1EnvCfg()
    snapshots = [parent.to_dict(), sibling.to_dict()]
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    ppo = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    assert type(cfg).__bases__ == (FinalGoalAccuracyEnvCfg,)
    child_dict = cfg.to_dict()
    changed = [key for key in snapshots[0] if snapshots[0][key] != child_dict[key]]
    assert set(child_dict) == set(snapshots[0])
    assert set(changed) == {'scene', 'rewards', 'terminations', 'curriculum'}, changed
    for before, cls in zip(snapshots, [FinalGoalAccuracyEnvCfg, FinalGoalAccuracyV1EnvCfg]):
        assert before == cls().to_dict()
    parent_ppo = load_cfg_from_registry('Isaac-Inspire-Final-GoalAccuracy-v0', 'rsl_rl_cfg_entry_point').to_dict()
    assert ppo.to_dict() == {**parent_ppo, 'experiment_name': 'inspire_final_goal_accuracy_v0_1'}
    assert not any('GoalAccuacy' in name for name in gym.registry)
    for gray, white in [(10, 10), (0, 0), (3, 4), (10, 10)]:
        left, right = parent.scene.copy(), cfg.scene.copy()
        for scene in (left, right):
            scene.configure_population_capacity(gray, white, 20)
        cubes = [v for v in right.object_collection.rigid_objects['mass_cylinder'].spawn.assets_cfg
                 if getattr(v, 'kind', None)=='cube']
        assert len(cubes) == 14
        for variant in cubes:
            assert variant.cube_size == (.06, .08, .10)
            variant.cube_size = (.08, .08, .08)
        assert left.to_dict() == right.to_dict(), 'Scene differs beyond target cube dimensions'
    report = {'task': task, 'parent_unchanged': True, 'sibling_unchanged': True,
              'changed_config_groups': changed, 'scene_rebuild': 'passed 4 capacities',
              'ppo': 'identical except experiment_name', 'usd_assets': []}
    # Exercise real spawn and prestartup scaling, not just the vertex helper.
    omni.usd.get_context().new_stage()
    stage = sim.get_current_stage()
    template = _target_spawn()
    for row, scale in enumerate((.9, 1., 1.1)):
        path = f'/World/envs/env_{row}/mass_cylinder'
        asset = FinalShelfObjectCfg(kind='cube', cube_size=(.06, .08, .10),
            rigid_props=template.rigid_props, mass_props=template.mass_props,
            collision_props=template.collision_props)
        asset.func(path, asset, translation=(row*2.5, 0, 1.05))
    env = NS(num_envs=3, sim=NS(is_playing=lambda:False), cfg=NS(seed=42,
        target_bounding_box=False, scene=NS(object_collection=NS(rigid_objects={'mass_cylinder':None}))))
    for row, scale in enumerate((.9, 1., 1.1)):
        randomize_object_sizes(env, [row], scale_range=(scale, scale))
        bounds = env._final_object_bounds[row, 0]
        np.testing.assert_allclose(bounds[1]-bounds[0], np.array([.06,.08,.10])*scale, atol=1e-7)
        root = stage.GetPrimAtPath(f'/World/envs/env_{row}/mass_cylinder')
        mesh = stage.GetPrimAtPath(str(root.GetPath())+'/Geometry/Part_0')
        assert mesh.HasAPI(UsdPhysics.CollisionAPI)
        world = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render']).ComputeWorldBound(root).ComputeAlignedRange()
        assert abs(world.GetMin()[2]-1.05)<1e-7
        report['usd_assets'].append({'scale':scale, 'bounds':bounds.tolist(), 'bottom':world.GetMin()[2]})
    (output/'config_and_assets.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    if args.smoke:
        cfg.scene.num_envs, cfg.sim.device, cfg.seed = 2, args.device, 42
        cfg.log_dir = str(output)
        env = gym.make(task, cfg=cfg)
        raw = env.unwrapped
        obs, _ = env.reset(seed=42)
        assert {k:v.shape[-1] for k,v in obs.items()} == {'policy':57, 'obstacles':80, 'privileged':3}
        shape_snapshot = raw._final_object_bounds.copy()
        for iteration in (0, 6000, 12000):
            raw.set_learning_iteration(iteration)
            env.reset()
            np.testing.assert_array_equal(shape_snapshot, raw._final_object_bounds)
            for _ in range(5):
                obs, reward, _, _, _ = env.step(torch.zeros(2, 8, device=raw.device))
                assert torch.isfinite(reward).all()
        raw.episode_length_buf[:] = raw.max_episode_length
        raw.termination_manager.compute()
        a = raw.termination_manager.get_term('target_goal_success_at_timeout')
        b = raw.termination_manager.get_term('target_goal_miss_at_timeout')
        assert (a | b).all() and not (a & b).any()
        (output/'smoke.json').write_text(json.dumps({'reset_steps_curriculum_timeout':'passed'}))
        env.close()
finally:
    app.close()
