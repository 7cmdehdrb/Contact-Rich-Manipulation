"""Native Fast reset acceptance probe; requires working Isaac Sim/CUDA assets.

Keeps output outside logs/rsl_rl. --layout-samples 100000 exercises the actual
spawned geometry; CPU synthetic tests are intentionally not its substitute.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--layout-samples', type=int, default=100000)
parser.add_argument('--output', type=Path, default=Path('outputs/relative_ik_fast_validation/native'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
result = {}
try:
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs import (
        WrenchRefactorRelativeIKV0_2_2_2TestEnvCfg as Parent,
        WrenchRefactorRelativeIKV0_2_2_2FastTestEnvCfg as Child)
    from inspire_sweep_baseline.mdp.common.wrench_refactor_layout import CircleResetLayout
    parent_task = 'Inspire-Wrench-Refactor-Relative-IK-v0_2_2_2-Test'
    task = 'Inspire-Wrench-Refactor-Relative-IK-v0_2_2_2-Fast-Test'
    before = Parent().to_dict()
    cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
    assert Child.__bases__ == (Parent,)
    current = cfg.to_dict()
    expected = copy.deepcopy(before)
    for key in ('events','max_obstacles','placement_clearance','layout_cell_size','layout_geometry_epsilon',
                'reset_ik_backend','debug_initial_ik_bounds','max_gray_obstacles','max_white_obstacles',
                'gray_padding','red_margin','rear_offset'):
        expected[key] = current[key]
    assert current == expected
    assert Parent().to_dict() == before
    runner = load_cfg_from_registry(parent_task, 'rsl_rl_cfg_entry_point').to_dict()
    runner['experiment_name'] = 'inspire_wrench_refactor_relative_ik_v0_2_2_2-Fast-Test'
    assert load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point').to_dict() == runner
    result['full_config_and_parent_preservation'] = True
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 42
    cfg.log_dir = str(args.output)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    with torch.inference_mode():
        obs, _ = env.reset(seed=42)
        assert obs['policy'].shape == (args.num_envs,58)
        assert obs['obstacles'].shape == (args.num_envs,64)
        assert raw.event_manager.active_terms['reset'] == ['object_spawn']
        assert not hasattr(raw, '_initial_ik_bounds_marker')
        ids = torch.arange(0,args.num_envs,2,device=raw.device)
        keep = torch.arange(1,args.num_envs,2,device=raw.device)
        robot, objects = raw.scene['robot'], raw.scene['object_collection']
        old_q = robot.data.joint_pos.clone()
        old_state = objects.data.object_link_state_w.clone()
        old_mass = objects.root_physx_view.get_masses().clone()
        old_inertia = objects.root_physx_view.get_inertias().clone()
        old_obs = raw._initial_obstacle_observation.clone()
        raw._reset_idx(ids)
        torch.testing.assert_close(robot.data.joint_pos[keep],old_q[keep])
        torch.testing.assert_close(objects.data.object_link_state_w[keep],old_state[keep])
        torch.testing.assert_close(raw._initial_obstacle_observation[keep],old_obs[keep])
        torch.testing.assert_close(objects.root_physx_view.get_masses(),old_mass)
        torch.testing.assert_close(objects.root_physx_view.get_inertias(),old_inertia)
        result['native_partial_reset'] = True
        generator = torch.Generator(device=raw.device).manual_seed(2717)
        histogram = torch.zeros(7,device=raw.device,dtype=torch.long)
        completed = 0
        while completed < args.layout_samples:
            size = min(args.num_envs,args.layout_samples-completed)
            selected = torch.arange(size,device=raw.device)
            p,y,a,d,n = raw._fast_layout.sample(selected,generator)
            assert torch.equal(a[:,1:].sum(-1),n)
            off = raw._fast_layout.offset[selected]
            c = p[...,:2]+torch.stack((y.cos()*off[...,0]-y.sin()*off[...,1],
                                      y.sin()*off[...,0]+y.cos()*off[...,1]),-1)
            radii = raw._fast_layout.expanded[selected]
            mask = a.clone();mask[:,0]=False
            assert ((c[...,0].abs()+radii<=.200001)|~mask).all()
            assert ((c[...,1].abs()+radii<=.500001)|~mask).all()
            pairs = mask[:,:,None]&mask[:,None,:]&~torch.eye(17,device=raw.device,dtype=torch.bool)[None]
            assert ((torch.cdist(c,c)>=radii[:,:,None]+radii[:,None,:]-1e-6)|~pairs).all()
            histogram += n.bincount(minlength=7)
            completed += size
        result['actual_geometry_samples'] = completed
        result['count_histogram'] = histogram.tolist()
        action = torch.zeros((args.num_envs,8),device=raw.device);action[:,-2:]=1.
        for step in range(180):
            if step == 90:
                raw.episode_length_buf[:] = raw.max_episode_length
            obs,reward,terminated,truncated,info = env.step(action)
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(reward).all()
        result['rollout_steps'] = 180
    env.close()
except BaseException:
    result['error'] = traceback.format_exc()
    raise
finally:
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    app.close()
