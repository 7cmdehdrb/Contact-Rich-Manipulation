"""Real grouped geometry/reset/physics and parent contract validation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.wrench_refactor_relative_ik_test_envs import (
        WrenchRefactorRelativeIKV0_2_1TestEnvCfg as Parent,
        WrenchRefactorRelativeIKV0_2_1ResetTestEnvCfg as Child)
    from isaaclab_tasks.utils import load_cfg_from_registry
    task = 'Inspire-Wrench-Refactor-Relative-IK-v0_2_1-Reset-Test'
    before = Parent().to_dict()
    default = load_cfg_from_registry(task, 'env_cfg_entry_point')
    current = default.to_dict()
    expected = copy.deepcopy(before)
    expected['events'] = current['events']
    for name in ('reset_size_groups','reset_pool_capacity','reset_pool_refresh_interval'):
        expected[name] = current[name]
    for obj in expected['scene']['object_collection']['rigid_objects'].values():
        obj['spawn']['reset_scale_groups'] = 16
    assert current == expected, [k for k in current if current[k] != expected.get(k)]
    assert Child.__bases__ == (Parent,)
    assert Parent().to_dict() == before
    parent_runner = load_cfg_from_registry('Inspire-Wrench-Refactor-Relative-IK-v0_2_1-Test', 'rsl_rl_cfg_entry_point').to_dict()
    parent_runner['experiment_name'] = 'inspire_wrench_refactor_relative_ik_v0_2_1-Reset-Test'
    assert load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point').to_dict() == parent_runner
    cfg = Child(reset_size_groups=2, reset_pool_capacity=2, reset_pool_refresh_interval=3)
    cfg.scene.num_envs = 8
    cfg.sim.device = args.device
    cfg.seed = 42
    out = Path('outputs/relative_ik_reset_test_validation')
    out.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(out)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    with torch.inference_mode():
        obs,_ = env.reset(seed=42)
        assert obs['policy'].shape == (8,58)
        for g in (0,1):
            scales = torch.as_tensor(raw._final_object_scales[g::2])
            torch.testing.assert_close(scales, scales[:1].expand_as(scales))
        assert not torch.equal(torch.as_tensor(raw._final_object_scales[0]),torch.as_tensor(raw._final_object_scales[1]))
        assert raw.event_manager.active_terms['reset'] == ['reset_all','object_spawn','reset_physics']
        assert 'success' not in raw.termination_manager.active_terms
        # Fill both slots, then prove the next reset does not invoke layout/IK.
        raw._reset_idx(torch.arange(8, device=raw.device))
        bank = raw._grouped_reset_pool
        generated = bank.generated
        import inspire_sweep_baseline.mdp.event_functions.wrench_refactor as parent_events
        old_layout, old_ik = parent_events.reset_scene, parent_events.reset_robot_from_goal_ik
        def forbidden(*a, **kw):
            raise AssertionError('Cache hit unexpectedly regenerated layout or IK')
        parent_events.reset_scene = parent_events.reset_robot_from_goal_ik = forbidden
        ids = torch.tensor([0,2,4], device=raw.device)
        untouched = torch.tensor([1,3,5,6,7], device=raw.device)
        objects = raw.scene['object_collection']
        old_state = objects.data.object_link_state_w.clone()
        old_q = raw.scene['robot'].data.joint_pos.clone()
        old_mass = objects.root_physx_view.get_masses().clone()
        old_inertia = objects.root_physx_view.get_inertias().clone()
        old_material = [m.view.get_material_properties().clone() for m in raw._clutter_physics.materials]
        raw._reset_idx(ids)
        parent_events.reset_scene, parent_events.reset_robot_from_goal_ik = old_layout, old_ik
        assert bank.generated == generated
        torch.testing.assert_close(objects.data.object_link_state_w[untouched],old_state[untouched])
        torch.testing.assert_close(raw.scene['robot'].data.joint_pos[untouched],old_q[untouched])
        state = objects.data.object_link_state_w[ids].clone()
        state[..., :3] -= raw.scene.env_origins[ids,None]
        torch.testing.assert_close(state,state[:1].expand_as(state),atol=1e-5,rtol=1e-5)
        q = raw.scene['robot'].data.joint_pos[ids]
        torch.testing.assert_close(q,q[:1].expand_as(q))
        rows = (torch.arange(17)[:,None]*8+ids.cpu()[None]).flatten()
        keep = (torch.arange(17)[:,None]*8+untouched.cpu()[None]).flatten()
        mass = objects.root_physx_view.get_masses()
        inertia = objects.root_physx_view.get_inertias()
        assert not torch.equal(mass[rows],old_mass[rows])
        torch.testing.assert_close(mass[keep],old_mass[keep])
        torch.testing.assert_close(inertia[keep],old_inertia[keep])
        physical = raw._clutter_physics
        torch.testing.assert_close(inertia[rows],physical.default_inertia[rows]*(mass[rows]/physical.default_mass[rows]))
        assert ((mass[rows] >= .230*1.5) & (mass[rows] <= .500*1.5)).all()
        for role,m in enumerate(physical.materials):
            props = m.view.get_material_properties()
            chosen = rows if role==1 else ids.cpu()
            unchosen = keep if role==1 else untouched.cpu()
            assert not torch.equal(props[chosen],old_material[role][chosen])
            torch.testing.assert_close(props[unchosen],old_material[role][unchosen])
        # Fill/reuse until refresh; ensure no permanent fixed initial state.
        for _ in range(4): raw._reset_idx(ids)
        assert bank.generated > generated
        raw.set_learning_iteration(10000)
        raw._reset_idx(ids)
        assert len(bank.entries[0]) == 1 and bank.entries[1] == []
        raw.set_learning_iteration(0)
        raw._reset_idx(torch.arange(8,device=raw.device))
        assert all(len(entries)==1 for entries in bank.entries)
        action = torch.zeros((8,8),device=raw.device);action[:,6:] = 1.
        for _ in range(30):
            obs,reward,*_ = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
        # A real timeout runs all reset events, including physics refresh.
        raw.episode_length_buf[:] = raw.max_episode_length-1
        obs,reward,terminated,truncated,extras = env.step(action)
        assert truncated.all()
        assert len([k for k in extras['log'] if k.startswith('Metrics/')]) == 8
        result = dict(task=task, full_config_comparison=True, parent_preserved=True,
                      shared_geometry=True, cache_hit_skips_layout_and_ik=True,
                      partial_reset_isolation=True, physics_redrawn_each_reset=True,
                      inertia_from_original_geometry=True, periodic_refresh=True,
                      curriculum_and_rewind_invalidation=True, real_timeout=True,
                      num_envs=8, rollout_steps=31)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    env.close()
except BaseException:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
    app.close()
