"""Actual config isolation and optional GPU reset/step validation for v0_2_1."""
import argparse
from pathlib import Path
import sys
import json
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility
parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--output-dir',type=Path,default=Path('outputs/final_accuracy_v0_2_1_validation'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
apply_renderer_compatibility(args)
app=AppLauncher(args).app
try:
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.mdp.reward_functions.final import target_contact_only
    from inspire_sweep_baseline.mdp.reward_functions.goal_rewards import success_event
    prefix='Isaac-Inspire-Final-GoalAccuracy-'
    parents={v:load_cfg_from_registry(prefix+v,'env_cfg_entry_point').to_dict() for v in ('v0','v0_1','v0_2','v0_3','v1')}
    cfg=load_cfg_from_registry(prefix+'v0_2_1','env_cfg_entry_point')
    parent=load_cfg_from_registry(prefix+'v0_2','env_cfg_entry_point')
    assert type(cfg).__bases__==(type(parent),)
    expected=parent.to_dict()
    actual=cfg.to_dict()
    for key in ('scene','events','rewards'):
        expected[key]=actual[key]
    assert expected==actual
    expected=parent.rewards.to_dict()
    for key in ('target_contact','target_goal_timeout_bonus'):
        expected[key]=actual['rewards'][key]
    assert expected==actual['rewards']
    assert list(parent.rewards.__dict__)==list(cfg.rewards.__dict__)
    assert cfg.rewards.target_contact.func is target_contact_only
    assert cfg.rewards.target_contact.weight==1.45
    assert cfg.rewards.target_goal_timeout_bonus.func is success_event
    assert cfg.rewards.target_goal_timeout_bonus.weight==75
    expected=parent.events.to_dict()
    expected['object_spawn']=cfg.events.object_spawn.to_dict()
    assert expected==cfg.events.to_dict()
    assert cfg.events.object_spawn.params==parent.events.object_spawn.params
    objects=cfg.scene.object_collection.rigid_objects
    assert len(objects)==36
    for name,obj in parent.scene.object_collection.rigid_objects.items():
        if name!='mass_cylinder':assert obj.to_dict()==objects[name].to_dict()
    for v,snapshot in parents.items():
        assert load_cfg_from_registry(prefix+v,'env_cfg_entry_point').to_dict()==snapshot
    ppo=load_cfg_from_registry(prefix+'v0_2_1','rsl_rl_cfg_entry_point').to_dict()
    old=load_cfg_from_registry(prefix+'v0_2','rsl_rl_cfg_entry_point').to_dict()
    old['experiment_name']='inspire_final_goal_accuracy_v0_2_1'
    assert ppo==old
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'config.json').write_text(json.dumps({'config':'passed','parent_isolation':list(parents),'physical_objects':len(objects),'logical_objects':21},indent=2))
    print('V0_2_1_CONFIG_PASS',flush=True)
    if args.smoke:
        cfg.scene.num_envs=2
        cfg.sim.device=args.device
        cfg.seed=42
        cfg.log_dir=str(args.output_dir)
        env=gym.make(prefix+'v0_2_1',cfg=cfg)
        raw=env.unwrapped
        obs,_=env.reset(seed=42)
        assert {k:v.shape[-1] for k,v in obs.items()}=={'policy':57,'obstacles':80,'privileged':3}
        seen=set()
        durations=[]
        for i in range(40):
            raw.set_learning_iteration((0,4000,4500,5000)[i%4])
            before=raw.scene['object_collection'].data.object_link_state_w[1].clone()
            kind=raw.target_kind_id[1].clone()
            start=time.perf_counter()
            raw._reset_idx(torch.tensor([0],device=raw.device))
            durations.append(time.perf_counter()-start)
            assert raw.target_kind_id[1]==kind
            torch.testing.assert_close(raw.scene['object_collection'].data.object_link_state_w[1],before)
            seen.add(int(raw.target_kind_id[0]))
            for _ in range(3):
                obs,reward,_,_,_=env.step(torch.zeros(2,8,device=raw.device))
                assert torch.isfinite(reward).all()
            logical=raw.scene['object_collection']
            physical=logical.physical
            unused=torch.ones((2,physical.num_objects),device=raw.device,dtype=torch.bool)
            unused[:,:21]=False
            unused[:,0]=True
            unused[logical.env_rows,logical.mapping[:,0]]=False
            assert (physical.data.object_pos_w[...,2][unused]>99).all()
        assert len(seen)>3
        # Force every candidate once; random smoke alone cannot guarantee coverage.
        sampler=raw._target_choice.sample
        logical=raw.scene['object_collection']
        from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
        for candidate in range(len(logical.pool_ids)):
            raw._target_choice.sample=lambda count,c=candidate:__import__('numpy').full(count,c,dtype='int64')
            env.reset()
            assert (raw.target_kind_id==candidate).all()
            torch.testing.assert_close(logical.data.object_pos_w[:,0],
                logical.physical.data.object_pos_w[logical.env_rows,logical.mapping[:,0]])
            _,reward,_,_,_=env.step(torch.zeros(2,8,device=raw.device))
            assert torch.isfinite(reward).all()
            for index,physical_id in enumerate(logical.pool_ids.cpu().tolist()):
                if index==candidate:continue
                # Read the original physical sensor, not the selected-target facade.
                sensor=(raw.scene.sensors[object_hand_sensor_name('mass_cylinder')].sensors[0]
                        if physical_id==0 else raw.scene[object_hand_sensor_name(logical.physical.object_names[physical_id])])
                assert not sensor.data.force_matrix_w.any()
        raw._target_choice.sample=sampler
        report={'reset_step':'passed','selected_candidates':sorted(seen),'partial_resets':40,'reset_mean_s':sum(durations)/len(durations)}
        report['forced_candidates']=len(logical.pool_ids)
        report['unused_candidate_hand_contact']='zero for all forced candidates'
        (args.output_dir/'smoke.json').write_text(json.dumps(report,indent=2))
        print('V0_2_1_SMOKE_PASS',report,flush=True)
        env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
