"""Live Hard reset contracts, full parent config comparison and layout benchmark."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--resets', type=int, default=20)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--output', type=Path, default=Path('outputs/prof_revision_hard_v0_3_validation/live.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app


def validate():
    import gymnasium as gym
    import numpy as np
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs.prof_revision_envs import ProfRevisionV0_3EnvCfg
    from inspire_sweep_baseline.assets import revision_spawners
    from inspire_sweep_baseline.mdp.common.wrench_refactor_layout import CircleResetLayout
    from inspire_sweep_baseline.mdp.common.prof_revision_layout import ProfRevisionHardLayout
    from inspire_sweep_baseline.mdp.common.stabilize_sweep_layout import configure_test_layout_batch
    from types import SimpleNamespace
    task = 'Inspire-Sweep-Prof-Revision-Hard-v0_3'
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    parent = ProfRevisionV0_3EnvCfg()
    # Compare against the actual pre-change source, independently of the child.
    import subprocess
    import types
    original = types.ModuleType('inspire_sweep_baseline.envs._pre_hard_prof_revision')
    original.__package__ = 'inspire_sweep_baseline.envs'
    sys.modules[original.__name__] = original
    original_source = subprocess.check_output(['git','show',
        'HEAD:src/inspire_sweep_baseline/inspire_sweep_baseline/envs/prof_revision_envs.py'],text=True)
    exec(compile(original_source,'<pre-Hard Prof Revision EnvCfg>','exec'),original.__dict__)
    assert type(cfg).__bases__ == (ProfRevisionV0_3EnvCfg,)
    a, b = parent.to_dict(), cfg.to_dict()
    def differences(a, b, path=''):
        result = []
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a.keys() | b.keys():
                result += differences(a.get(key), b.get(key), f'{path}.{key}'.strip('.'))
        elif a != b:
            result.append((path, str(a), str(b)))
        return result
    original_diff = differences(original.ProfRevisionV0_3EnvCfg().to_dict(), parent.to_dict())
    assert not original_diff, ('parent config drift', original_diff)
    diff = differences(a, b)
    allowed = {'scene.object_collection.rigid_objects.mass_cylinder.spawn.kind_override',
               'events.object_geometry.params.layout_class', 'events.object_spawn.func',
               'events.object_spawn.params.min_goal_distance', 'events.object_spawn.params.max_goal_distance',
               'max_obstacles', 'layout_cell_size'}
    assert {d[0] for d in diff} <= allowed, diff
    ppo = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point').to_dict()
    base_ppo = load_cfg_from_registry('Inspire-Sweep-Prof-Revision-v0_3', 'rsl_rl_cfg_entry_point').to_dict()
    assert ppo.pop('experiment_name') == 'inspire_sweep_prof_revision_hard_v0_3'
    base_ppo.pop('experiment_name')
    assert ppo == base_ppo
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    cfg.log_dir = str(args.output.parent.resolve())
    env = gym.make(task, cfg=cfg).unwrapped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            np.save(args.output.parent/'bounds.npy', env._revision_bounds_t.cpu().numpy())
            obs, _ = env.reset()
            assert obs['policy'].shape == (args.num_envs,108)
            assert obs['obstacles'].shape == (args.num_envs,176)
            bounds = env._revision_bounds_t
            np.save(args.output.parent/'bounds.npy', bounds.cpu().numpy())
            kinds = [revision_spawners._prototype_cache['metadata'][f'/World/envs/env_{i}/mass_cylinder'][0]
                     for i in range(args.num_envs)]
            if args.num_envs >= 64:
                assert len(set(kinds)) == 8
            hist = torch.zeros(11, device=env.device, dtype=torch.long)
            distances = []
            def check_layout(ids):
                layout = env._fast_layout
                # Exercise the cached sampler independently, including partial IDs.
                local, yaw, active, delta, requested = layout.sample(ids, env._revision_layout_rng,
                    **{k:v for k,v in cfg.events.object_spawn.params.items()
                       if k in ('spawn_radius','edge_clearance','excluded_x_half_angle','min_goal_distance','max_goal_distance')})
                assert torch.equal(active[:,1:].sum(-1), requested)
                assert bool(((requested >= 0) & (requested <= 10)).all())
                hist.add_(torch.bincount(requested, minlength=11))
                dist = delta.norm(dim=-1)
                assert bool((dist >= .05-1e-6).all())
                distances.extend(dist.cpu().tolist())
                off = layout.offset[ids]
                rot = torch.stack((yaw.cos()*off[...,0]-yaw.sin()*off[...,1],
                                   yaw.sin()*off[...,0]+yaw.cos()*off[...,1]),-1)
                centers = local[...,:2]+rot
                half = (bounds[ids,:,1,:2]-bounds[ids,:,0,:2])/2
                ext = torch.stack((yaw.cos().abs()*half[...,0]+yaw.sin().abs()*half[...,1],
                                   yaw.sin().abs()*half[...,0]+yaw.cos().abs()*half[...,1]),-1)
                assert bool(((centers.abs()+ext <= bounds.new_tensor([.2,.5])+1e-6) | ~active[...,None]).all())
                goal_center = centers[:,0]+delta
                assert bool((goal_center.abs()+ext[:,0] <= bounds.new_tensor([.18,.48])+1e-6).all())
                separation = (centers[:,:,None]-centers[:,None,:]).abs()
                limits = ext[:,:,None]+ext[:,None,:]+cfg.placement_clearance
                pairs = active[:,:,None] & active[:,None,:] & ~torch.eye(17,device=env.device,dtype=torch.bool)[None]
                assert bool((separation >= limits-1e-6).any(-1)[pairs].all())
            ids = torch.arange(env.num_envs,device=env.device)
            for reset_index in range(args.resets):
                check_layout(ids)
                env._reset_idx(ids)
                assert torch.equal(env._active_object_mask[:,1:].sum(-1),env._final_actual_population)
                assert bool(torch.isfinite(env._obb_target_size).all() & (env._obb_target_size>0).all())
                assert bool((env._final_actual_population<=10).all())
                if reset_index % 10 == 0:
                    print(f'Validated reset batch {reset_index+1}/{args.resets}',flush=True)
            # A partial physical reset must leave every other rigid state unchanged.
            before = env.scene['object_collection'].data.object_link_state_w.clone()
            env._reset_idx(ids[::2])
            torch.testing.assert_close(env.scene['object_collection'].data.object_link_state_w[1::2],before[1::2],rtol=0,atol=0)
            done_count = 0
            for _ in range(args.steps):
                obs, reward, terminated, truncated, _ = env.step(torch.zeros((env.num_envs,12),device=env.device))
                assert bool(torch.isfinite(reward).all())
                assert all(bool(torch.isfinite(v).all()) for v in obs.values())
                done_count += int((terminated|truncated).sum())
            # Same actual asset bounds, 2048 rows; warm compilation excluded.
            bench_bounds = bounds.repeat((2048+len(bounds)-1)//len(bounds),1,1,1)[:2048]
            spawn_z = env._fast_layout.spawn_z.repeat((2048+len(bounds)-1)//len(bounds),1)[:2048]
            rows = torch.arange(2048,device=env.device)
            times = {}
            for name, cls, cell, params in (
                ('parent',CircleResetLayout,.005,dict(min_goal_distance=.06,max_goal_distance=.12,excluded_x_half_angle=0.)),
                ('hard',ProfRevisionHardLayout,.01,dict(min_goal_distance=.05,max_goal_distance=None,excluded_x_half_angle=0.))):
                layout = cls(bench_bounds,spawn_z,cell_size=cell)
                if name=='parent':
                    configure_test_layout_batch(SimpleNamespace(_fast_layout=layout))
                rng = torch.Generator(device=env.device).manual_seed(123)
                layout.sample(rows,rng,**params)
                torch.cuda.synchronize()
                samples=[]
                for _ in range(10):
                    start=perf_counter()
                    layout.sample(rows,rng,**params)
                    torch.cuda.synchronize()
                    samples.append(1000*(perf_counter()-start))
                times[name]={'median_ms':float(np.median(samples)),'samples_ms':samples}
            result = dict(config_differences=diff,parent_matches_prechange=True,ppo_equal=True,target_kind_histogram=np.bincount(kinds,minlength=8).tolist(),
                          obstacle_count_histogram=hist.cpu().tolist(),minimum_goal_m=min(distances),maximum_goal_m=max(distances),
                          layout_benchmark_2048=times,steps=args.steps,completed_episodes=done_count,
                          reset_batches=args.resets,observation_shapes={k:list(v.shape) for k,v in obs.items()})
            args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,indent=2),flush=True)
    finally:
        env.close()


try:
    validate()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    app.close()
