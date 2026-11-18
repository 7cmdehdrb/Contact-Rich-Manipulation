"""GPU integration/audit for Inspire-Stabilize-Sweep-v0 (no training logs written).

Run with the Isaac Lab Python interpreter and --headless. Sensor replay is
reported separately from real physics; it does not prove Gate attainability.
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audit-resets',type=int,default=8)
parser.add_argument('--output',type=Path,default=Path('/tmp/stabilize_sweep_validation.json'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app


def validate():
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab.utils.math import quat_apply,quat_apply_inverse
    from inspire_sweep_baseline.mdp.common.stabilize_sweep_contact import centroid
    cfg=load_cfg_from_registry('Inspire-Stabilize-Sweep-v0','env_cfg_entry_point')
    cfg.seed=42;cfg.scene.num_envs=2
    cfg.scene.ee_frame.debug_vis=False;cfg.commands.target_goal_pos.debug_vis=False
    env=gym.make('Inspire-Stabilize-Sweep-v0',cfg=cfg).unwrapped
    result={'seed':42,'num_envs':2,'reset_audit':[]}
    try:
        # Audit the specified distribution. Failed commands are reported, not redrawn inside reset.
        for seed in range(args.audit_resets):
            start=perf_counter()
            try:
                obs,_=env.reset(seed=seed)
                torch.cuda.synchronize()
                result['reset_audit'].append(dict(seed=seed,ok=True,ms=1000*(perf_counter()-start),
                    metrics={k:float(v) for k,v in env.extras['log'].items() if k.startswith('Reset/')}))
            except RuntimeError as error:
                if 'exhausted 4 candidates' not in str(error):raise
                result['reset_audit'].append(dict(seed=seed,ok=False,error=str(error),ms=1000*(perf_counter()-start)))
        # Separate selected-row resets use an explicit reproducible fixture seed.
        for row in range(2):obs,_=env.reset(seed=42,env_ids=torch.tensor([row],device=env.device))
        assert obs['policy'].shape==(2,67) and obs['obstacles'].shape==(2,64)
        target=env._surface_target_w.clone();points=env._surface_obstacles_w.clone()
        center=centroid(env);data=env.scene['object_collection'].data
        offset=env._revision_bounds_t[:,0].mean(1)
        torch.testing.assert_close(center,data.object_link_pos_w[:,0]+quat_apply(data.object_link_quat_w[:,0],offset))
        result['centroid_com']={'height':(env._revision_bounds_t[:,0,1,2]-env._revision_bounds_t[:,0,0,2]).tolist(),
            'centroid_minus_root':(center-data.object_link_pos_w[:,0]).tolist(),
            'com_minus_root':(data.object_com_pos_w[:,0]-data.object_link_pos_w[:,0]).tolist()}
        held=env.scene['object_collection'].data.object_link_state_w[1].clone()
        joints=env.scene['robot'].data.joint_pos[1].clone();clock=env._sim_step_counter
        env.reset(seed=42,env_ids=torch.tensor([0],device=env.device))
        assert env._sim_step_counter==clock
        torch.testing.assert_close(held,env.scene['object_collection'].data.object_link_state_w[1],atol=0,rtol=0)
        torch.testing.assert_close(joints,env.scene['robot'].data.joint_pos[1],atol=0,rtol=0)
        torch.testing.assert_close(target[1],env._surface_target_w[1],atol=0,rtol=0)
        result['partial_reset_isolation']=True
        start=perf_counter()
        for _ in range(4):
            obs,reward,terminated,truncated,_=env.step(torch.zeros((2,12),device=env.device))
            assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
            assert not (terminated|truncated).any()
        torch.cuda.synchronize();result['real_step_ms']=1000*(perf_counter()-start)/4
        assert env._stabilize_contacts.samples==2
        torch.testing.assert_close(env._surface_target_w,target)
        torch.testing.assert_close(env._surface_obstacles_w,points)
        frame=env.scene['ee_frame'].data
        expected=quat_apply_inverse(frame.target_quat_w[:,0],target-frame.target_pos_w[:,0])
        # Direct observation term output, with the full registered manager also exercised above.
        from inspire_sweep_baseline.mdp.observation_functions.stabilize_sweep import target_surface_in_eef
        torch.testing.assert_close(target_surface_in_eef(env),expected)
        result['real_physics_smoke']=True
        # Keep physical step/reset code live; replace only state-sample inputs with known valid loads.
        # This checks the Manager phase boundary, not physical attainability of those loads.
        state=env._stabilize_state;sample=state.sample
        original_center=state.initial_center.clone();original_quat=state.initial_quat.clone()
        goal=original_center+torch.tensor([.1,0.,0.],device=env.device)
        def replay(*unused):
            sample(original_center,original_quat,goal,
                   torch.tensor([[.5,0.,0.]],device=env.device).expand(2,-1),
                   torch.zeros((2,3),device=env.device),torch.full((2,),.5,device=env.device),
                   torch.ones(2,device=env.device,dtype=torch.bool),torch.zeros(2,device=env.device),torch.zeros(2,device=env.device))
        state.sample=replay
        try:
            for step in range(15):
                obs,reward,terminated,truncated,_=env.step(torch.zeros((2,12),device=env.device))
                assert not (terminated|truncated).any()
                assert bool((obs['policy'][:,-1]==(1. if step==14 else 0.)).all())
            assert not state.reward_phase.any() and state.achieved.all()
            obs,*_=env.step(torch.zeros((2,12),device=env.device))
            assert state.reward_phase.all() and (state.stage_score==1.).all()
            result['controlled_gate_manager_boundary']=True
            # Accelerated terminal counter: same-step Gate latch counts at timeout,
            # and only that row resets while the other stays in Sweep.
            state.achieved[0]=False;state.consecutive[0]=28
            env.episode_length_buf[0]=env.max_episode_length-1
            env.command_manager.get_term('target_goal_pos').pos_command_w[0,:2]=centroid(env)[0,:2]+.005
            env._revision_layout_rng.manual_seed(42);env._goal_reset_rng.manual_seed(42+1707)
            before=int(env._timeout_outcomes.totals[0])
            obs,reward,terminated,truncated,_=env.step(torch.zeros((2,12),device=env.device))
            assert truncated.tolist()==[True,False] and not terminated.any()
            assert int(env._timeout_outcomes.totals[0])==before+1
            assert obs['policy'][:,-1].tolist()==[0.,1.]
            result['same_step_gate_timeout_and_partial_reset']=True
        finally:state.sample=sample
        # Controlled geometry probe: move the cylinder toward the hand until actual
        # contact exists, then compare packed point reductions with native force readback.
        objects=env.scene['object_collection'];initial=objects.data.object_link_state_w.clone()
        direction=env.command_manager.get_command('target_goal_pos')[:,:3]-centroid(env)
        direction[:,2]=0.;direction/=direction.norm(dim=-1,keepdim=True)
        measured=False
        palm=env.scene['robot'].find_bodies('inspire_palm_force_sensor')[0][0]
        palm_position=env.scene['robot'].data.body_link_pos_w[:,palm].clone()
        radius=env._revision_bounds_t[:,0,:,0].abs().amax(-1)
        for fraction in (1.1,1.,.9,.7,.5,0.):
            pose=initial.clone()
            pose[:,0,:3]=palm_position+fraction*radius[:,None]*direction-quat_apply(pose[:,0,3:7],offset)
            pose[:,0,7:]=0.
            objects.write_object_link_state_to_sim(pose)
            env.action_manager.apply_action();env.scene.write_data_to_sim()
            env.sim.step(render=False);env.scene.update(env.physics_dt)
            contacts=env._stabilize_contacts;contacts.capture(env)
            if not bool((contacts.load>.01).any()):continue
            view=contacts.view
            normal=tuple(v.clone() for v in view.get_contact_data(dt=env.physics_dt))
            friction=tuple(v.clone() for v in view.get_friction_data(dt=env.physics_dt))
            scalar,positions,normals,_,counts,starts=(v.cpu() for v in normal)
            tangent,anchors,fcounts,fstarts=(v.cpu() for v in friction)
            com=objects.data.object_com_pos_w[:,0].cpu()
            normal_sum=torch.zeros(2,3);total=torch.zeros(2,3);torque=torch.zeros(2,3);load=torch.zeros(2)
            for row in range(2):
                for pair in range(counts.shape[1]):
                    for k in range(int(starts[row,pair]),int(starts[row,pair]+counts[row,pair])):
                        f=scalar.reshape(-1)[k]*normals[k]
                        normal_sum[row]+=f;total[row]+=f;torque[row]+=torch.cross(positions[k]-com[row],f,dim=-1);load[row]+=f.norm()
                    for k in range(int(fstarts[row,pair]),int(fstarts[row,pair]+fcounts[row,pair])):
                        total[row]+=tangent[k];torque[row]+=torch.cross(anchors[k]-com[row],tangent[k],dim=-1)
            native=view.get_contact_force_matrix(dt=env.physics_dt).sum(1).cpu()
            torch.testing.assert_close(normal_sum,native,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(contacts.force.cpu(),total,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(contacts.moment.cpu(),torque,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(contacts.load.cpu(),load,atol=1e-4,rtol=1e-4)
            result['actual_contact_probe']={'load_N':load.tolist(),'force_N':total.tolist(),'moment_Nm':torque.tolist()}
            measured=True;break
        assert measured,'No physical hand/target contact found in the controlled probe'
        result['final_reset_metrics']={k:float(v) for k,v in env.extras['log'].items() if k.startswith('Reset/')}
    finally:
        args.output.write_text(json.dumps(result,indent=2))
        env.close()
    print('STABILIZE_VALIDATION_PASS',args.output,flush=True)


try:
    validate()
except Exception:
    import traceback,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
    # Kit's stop callback can wait forever after an initialization exception.
    os._exit(1)
finally:
    app.close()
