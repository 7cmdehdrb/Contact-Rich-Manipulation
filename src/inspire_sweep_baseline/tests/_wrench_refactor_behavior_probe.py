"""GUI, 32-vector reset and rule-driven front/back push audit (no RL).

Push mode overrides only episode duration and deterministic zero-obstacle reset.
Manual stepping evaluates all production terms before reset, records terminal
rows once, and keeps the scene visible without automatic episode replacement.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['reset','push'],required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--resets-per-stage',type=int,default=32)
parser.add_argument('--pace',type=float,default=1.)
parser.add_argument('--hold-view-seconds',type=float,default=15.)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.headless:raise ValueError('This visual audit requires GUI; omit --headless')
args.output.mkdir(parents=True,exist_ok=True)
app=AppLauncher(args).app
import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul,quat_conjugate,axis_angle_from_quat,euler_xyz_from_quat
import inspire_sweep_baseline
from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorEnvCfg
from inspire_sweep_baseline.mdp.event_functions.wrench_refactor import reset_scene
from inspire_sweep_baseline.mdp.event_functions.wrench_refactor import apply_fixed_pool_gravity_mask
from inspire_sweep_baseline.mdp.common.revision_goal import object_geometry_in_shelf
from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w
from audit_wrench_refactor_sampling import direction_report


def fixed_push_reset(env,env_ids):
    reset_scene(env,env_ids)
    ids=torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    n=len(ids);objects=env.scene['object_collection'];shelf=env.scene['shelf'].data
    local=torch.zeros(n,17,3,device=env.device)
    local[...,2]=50+2*torch.arange(17,device=env.device)
    local[:,0,2]=env._refactor_spawn_z[ids,0]
    rotation=shelf.root_quat_w[ids,None].expand(-1,17,-1)
    world=shelf.root_pos_w[ids,None]+quat_apply(rotation,local)
    states=torch.cat([world,rotation,torch.zeros(n,17,6,device=env.device)],-1)
    objects.write_object_link_state_to_sim(states,env_ids=ids)
    env._active_object_mask[ids]=False;env._active_object_mask[ids,0]=True
    apply_fixed_pool_gravity_mask(env,ids)
    delta=torch.zeros(n,3,device=env.device);delta[:,1]=torch.where(ids<16,-.12,.12)
    env.sweep_dir[ids]=quat_apply(shelf.root_quat_w[ids],delta)
    env._initial_obstacle_observation[ids]=0
    env._final_actual_population[ids]=0
    env._revision_spawn_positions[ids]=world


def advance(env,action,evaluate=True):
    env.sensor_data_fresh[:]=True
    env.action_manager.process_action(action.to(env.device))
    if hasattr(env, '_probe_expected_pose'):
        term=env.action_manager.get_term('arm_action');root=env.scene['robot'].data
        pose=term._osc.desired_ee_pose_b
        world_p=root.root_pos_w+quat_apply(root.root_quat_w,pose[:,:3])
        world_q=quat_mul(root.root_quat_w,pose[:,3:])
        expected_p,expected_q=env._probe_expected_pose
        pe=float((world_p-expected_p).abs().max())
        qe=float((1-(world_q*expected_q).sum(-1).abs()).abs().max())
        env._probe_frame_error=max(getattr(env,'_probe_frame_error',0.),pe)
        assert pe<1e-5 and qe<1e-5,(pe,qe)

    for _ in range(env.cfg.decimation):
        env._sim_step_counter+=1
        env.action_manager.apply_action();env.scene.write_data_to_sim()
        env.sim.step(render=False);env.sim.render();env.scene.update(env.physics_dt)
    env.episode_length_buf+=1;env.common_step_counter+=1
    if evaluate:
        done=env.termination_manager.compute().clone()
        reward=env.reward_manager.compute(env.step_dt).clone()
        obs=env.observation_manager.compute()
        return done,reward,obs


def reset_audit(env):
    reports=[];all_deltas=[];worst_initial=0.;worst_after=0.;worst_jump=0.;parked_drift=0.
    maximum_speed=0.;observed_bad=[];timings=[]
    masses=env.scene['object_collection'].root_physx_view.get_masses().clone()
    materials=[m.properties.clone() for m in env._clutter_physics.materials]
    geometry=env._revision_bounds_t.clone()
    for stage in (0,10000,13000,16000,19000,22000,25000):
        env.set_learning_iteration(stage);angles=[]
        for trial in range(args.resets_per_stage):
            t=time.perf_counter();env.reset();torch.cuda.synchronize();timings.append(time.perf_counter()-t)
            local,bottom=object_geometry_in_shelf(env);active=env._active_object_mask.clone()
            initial=env.scene['object_collection'].data.object_pos_w.clone()
            gravity=env.scene['object_collection'].root_physx_view.get_disable_gravities().clone()
            expected=(~active).T.reshape(-1,1).cpu().to(torch.uint8)
            assert torch.equal(gravity.cpu(),expected), 'Gravity flags differ from requested active mask'
            worst_initial=max(worst_initial,float((bottom[active]-1.05).abs().max()))
            assert float((bottom[active]-1.05).abs().max())<2e-5
            assert torch.equal(env._final_actual_population.sum(-1)+1,active.sum(-1))
            delta=quat_apply_inverse(env.scene['shelf'].data.root_quat_w,env.sweep_dir)[:,:2].cpu().numpy()
            angles.append(delta)
            # Observe the first 100ms for every eighth reset; no observation or
            # auto-reset is allowed to hide a falling/teleported object.
            if trial%8==0:
                action=torch.zeros(32,8,device=env.device);action[:,-2:]=1.
                previous=initial
                for tick in range(5):
                    advance(env,action,False)
                    pos=env.scene['object_collection'].data.object_pos_w.clone()
                    _,z=object_geometry_in_shelf(env)
                    worst_after=max(worst_after,float((z[active]-1.05).abs().max()))
                    worst_jump=max(worst_jump,float((pos[...,2]-previous[...,2])[active].abs().max()))
                    drift=(pos-initial).abs().amax(-1).masked_fill(active,0)
                    maximum=float(drift.max())
                    if maximum>parked_drift+.01:
                        flat=int(drift.argmax());e,j=divmod(flat,17)
                        print('PARKING_DRIFT',json.dumps(dict(env=e,object=j,delta=(pos-initial)[e,j].tolist(),velocity=env.scene['object_collection'].data.object_lin_vel_w[e,j].tolist(),gravity_disabled=int(gravity[j*32+e,0]))),flush=True)
                    parked_drift=max(parked_drift,maximum)
                    maximum_speed=max(maximum_speed,float(env.scene['object_collection'].data.object_lin_vel_w[active].norm(dim=-1).max()))
                    if bool((z[active]<1.03).any()):observed_bad.append({'stage':stage,'reset':trial,'tick':tick,'min_bottom':float(z[active].min())})
                    previous=pos
            assert torch.equal(masses,env.scene['object_collection'].root_physx_view.get_masses())
            assert torch.equal(geometry,env._revision_bounds_t)
            assert all(torch.equal(old,m.properties) for old,m in zip(materials,env._clutter_physics.materials))
        degrees=max(0,30-5*(0 if stage<10000 else 1+(stage-10000)//3000))
        row=direction_report(np.concatenate(angles),degrees);reports.append(row);all_deltas.append(np.concatenate(angles))
        print('RESET_STAGE',json.dumps(row),flush=True)
    result=dict(mode='reset',num_envs=32,stages=reports,reset_samples=len(timings)*32,
        worst_initial_bottom_error_m=worst_initial,worst_first_100ms_bottom_error_m=worst_after,
        largest_active_z_step_m=worst_jump,parked_drift_m=parked_drift,max_active_speed_m_s=maximum_speed,
        below_surface_cases=observed_bad,median_reset_s=float(np.median(timings)),max_reset_s=max(timings))
    assert parked_drift<1e-4, f'Parked objects moved: {parked_drift}m'
    assert not observed_bad
    (args.output/'reset.json').write_text(json.dumps(result,indent=2));np.savez_compressed(args.output/'directions.npz',*all_deltas)


def move_action(env,desired,quat):
    pose=env.scene['ee_frame'].data
    p=pose.target_pos_w[:,0];q=pose.target_quat_w[:,0]
    error=quat_apply_inverse(q,desired-p)
    # Small physical increments; retain the production OSC and torque limits.
    error=error*torch.clamp(.03/error.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1)
    dq=quat_mul(quat_conjugate(q),quat)
    aa=axis_angle_from_quat(dq)
    if getattr(env,'_probe_orientation_pi',False):
        integral=getattr(env,'_probe_orientation_integral',torch.zeros_like(aa))
        integral=integral+quat_apply(q,aa)*env.step_dt
        integral=integral*torch.clamp(.20/integral.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1)
        env._probe_orientation_integral=integral
        aa=aa+quat_apply_inverse(q,integral)
    aa=aa*torch.clamp(.25/aa.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1)
    # The production UR10 action wrapper accepts RPY and converts to the
    # Isaac Lab controller's axis-angle representation in _preprocess_actions.
    from isaaclab.utils.math import quat_from_angle_axis
    small=quat_from_angle_axis(aa.norm(dim=-1),aa/aa.norm(dim=-1,keepdim=True).clamp_min(1e-9))
    rpy=torch.stack(euler_xyz_from_quat(small),-1);rpy=(rpy+torch.pi)%(2*torch.pi)-torch.pi
    action=torch.ones(32,8,device=env.device);action[:,:3]=error/.03;action[:,3:6]=rpy/.15
    env._probe_expected_pose=(p+quat_apply(q,error),quat_mul(q,small))
    return action


def hand_geometry_probe(env):
    """Cache link-local mesh box corners; transform with live PhysX link poses."""
    import itertools
    from pxr import Usd, UsdGeom
    robot=env.scene['robot']
    stage=Usd.Stage.Open(env.cfg.scene.robot.spawn.usd_path)
    root=str(stage.GetDefaultPrim().GetPath());cache=UsdGeom.XformCache()
    indices=[];corners=[]
    for name in robot.body_names:
        body=stage.GetPrimAtPath(root+'/'+name)
        if not body.IsValid():continue
        inv=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)
        for prim in Usd.PrimRange(body):
            if prim.GetName()!='right_geometry':continue
            vertices=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            tf=inv@np.asarray(cache.GetLocalToWorldTransform(prim)).T
            points=(np.c_[vertices,np.ones(len(vertices))]@tf.T)[:,:3]
            corners.append(list(itertools.product(*zip(points.min(0),points.max(0)))))
            indices.append(robot.body_names.index(name))
    points=torch.tensor(corners,device=env.device,dtype=torch.float32)[None].expand(32,-1,-1,-1)
    def bounds():
        q=robot.data.body_link_quat_w[:,indices,None].expand(-1,-1,8,-1)
        xyz=quat_apply(q,points)+robot.data.body_link_pos_w[:,indices,None]
        return torch.stack((xyz.amin((1,2)),xyz.amax((1,2))),1)
    return bounds


def push_audit(env):
    # Derived from the right-hand USD open pose; fingers -X, pad local +Z -> world -Y.
    calibration={'eef_quaternion_wxyz': [-0.03318236724565661, -0.03318520551391893, -0.70632517815725, 0.706330245239507], 'hand_bounds_from_eef_world': [[-0.11741735474238141, -0.05314858428794792, -0.05617525204679363], [0.13172438027080843, 0.035011515632481685, 0.1218669606875294]], 'palm_center_from_eef_world': [0.008320469031895953, -0.012553714300858195, 0.0011071632935052787], 'palm_normal_world': [-1.8653810245493453e-18, -1.0000000000000002, -3.2983688525312613e-22], 'finger_direction_world': [-0.9999999999999998, -1.8653810245493453e-18, -7.333136295124255e-18]}
    desired_q=torch.tensor(calibration['eef_quaternion_wxyz'],device=env.device).expand(32,-1)
    target=env.scene['object_collection'].data.object_pos_w[:,0].clone()
    initial=env.scene['ee_frame'].data.target_pos_w[:,0].clone()
    radius=env._revision_bounds_t[:,0,1,0]
    sign=torch.where(torch.arange(32,device=env.device)<16,-1.,1.)
    # Palm normal points world -Y for both cases; fingers point world -X.
    near=target.clone();near[:,0]-=calibration['palm_center_from_eef_world'][0]
    near[:,2]=env.scene['shelf'].data.root_pos_w[:,2]+1.13
    near[:,1]-=sign*(radius+.085)
    # The production default wrist now faces the palm left already. Lower at
    # the initial front position, align sideways, then enter horizontally.
    oriented=initial.clone();oriented[:,2]=near[:,2]
    aligned=oriented.clone();aligned[:,1]=near[:,1]
    contact=near.clone();contact[:,1]=target[:,1]-sign*(radius+torch.where(sign<0,.016,.035))
    end=contact.clone();end[:,1]+=sign*.145
    phases=[('baseline',2.,initial,initial),('orient_outside',4.,initial,oriented),
            ('align_outside',4.,oriented,aligned),('enter_horizontal',5.,aligned,near),
            ('approach_contact',3.,near,contact),('push',8.,contact,end),('hold',6.,end,end)]
    print('PUSH_INITIAL',json.dumps(dict(eef=initial[0].tolist(),target=target[0].tolist(),near=near[0].tolist(),end=end[0].tolist(),quaternion=desired_q[0].tolist())),flush=True)
    hand_bounds=hand_geometry_probe(env)
    reward_names=env.reward_manager.active_terms;done_names=env.termination_manager.active_terms
    records=[];terminated=torch.zeros(32,dtype=torch.bool,device=env.device)
    initial_reach_best=env.reward_manager.get_term_cfg('target_reaching').func._best_distance.cpu().tolist()
    initial_previous_action=env.action_manager.prev_action.cpu().tolist()
    initial_q=env.scene['ee_frame'].data.target_quat_w[:,0].clone()
    def save_trace():
        data={key:np.stack([r[key] for r in records]) for key in records[0] if key!='phase'}
        data['phase']=np.array([r['phase'] for r in records]);np.savez_compressed(args.output/'push_trace.tmp.npz',**data);(args.output/'push_trace.tmp.npz').replace(args.output/'push_trace.npz')
        (args.output/'channels.json').write_text(json.dumps(dict(rewards=reward_names,terminations=done_names,left_envs=list(range(16)),right_envs=list(range(16,32)),test_episode_s=40.,auto_reset=False,robot_joint_names=env.scene['robot'].joint_names,hand_velocity_joint_names=env.cfg.rewards.robot_joint_velocity_penalty.params['hand_cfg'].joint_names,orientation_integral_gain=1.,initial_reach_best=initial_reach_best,initial_previous_action=initial_previous_action,command_frame_max_error_m=getattr(env,"_probe_frame_error",None),calibration=calibration),indent=2))

    for phase,duration,start,finish in phases:
        count=round(duration/env.step_dt)
        for i in range(count):
            wall=time.perf_counter()
            u=(i+1)/count;s=u*u*(3-2*u)
            desired=start+(finish-start)*s
            quat=initial_q if phase in ('baseline','withdraw') else desired_q
            if phase=='orient_outside':
                from isaaclab.utils.math import quat_from_angle_axis
                rel=quat_mul(quat_conjugate(initial_q),desired_q)
                aa=axis_angle_from_quat(rel); angle=aa.norm(dim=-1)
                quat=quat_mul(initial_q,quat_from_angle_axis(angle*s,aa/angle[:,None].clamp_min(1e-8)))
            env._probe_orientation_pi=phase in ('approach_contact','push','hold')
            action=move_action(env,desired,quat)
            done,total,obs=advance(env,action)
            tactile_cfg=env.observation_manager.cfg.policy.tactile
            tactile=tactile_cfg.func(env,**tactile_cfg.params)
            wrench_cfg=env.observation_manager.cfg.policy.wrist_wrench
            wrench=wrench_cfg.func(env,**wrench_cfg.params)
            row=dict(phase=phase,time_s=env.common_step_counter*env.step_dt,
                position=env.scene['object_collection'].data.object_pos_w[:,0].cpu().numpy(),
                goal=env.command_manager.get_command('target_goal_pos')[:,:3].cpu().numpy(),
                eef=env.scene['ee_frame'].data.target_pos_w[:,0].cpu().numpy(),
                desired_eef=desired.cpu().numpy(),hand_bounds=hand_bounds().cpu().numpy(),
                tactile=tactile.cpu().numpy(),tactile_observed=obs['policy'][:,34:68].cpu().numpy(),
                wrench=wrench.cpu().numpy(),wrench_observed=obs['policy'][:,68:74].cpu().numpy(),
                contact_force=target_hand_contact_force_w(env).cpu().numpy(),
                palm_normal=quat_apply(env.scene['robot'].data.body_link_quat_w[:,env.scene['robot'].body_names.index('inspire_palm_force_sensor')],torch.tensor([0.,0.,1.],device=env.device).expand(32,-1)).cpu().numpy(),
                target_velocity=env.scene['object_collection'].data.object_lin_vel_w[:,0].cpu().numpy(),
                target_quaternion=env.scene['object_collection'].data.object_quat_w[:,0].cpu().numpy(),
                eef_quaternion=env.scene['ee_frame'].data.target_quat_w[:,0].cpu().numpy(),
                joint_velocity=env.scene['robot'].data.joint_vel.cpu().numpy(),
                joint_position=env.scene['robot'].data.joint_pos.cpu().numpy(),
                jacobian_sigma_min=torch.linalg.svdvals(env.action_manager.get_term('arm_action')._jacobian_b)[:,-1].cpu().numpy(),
                applied_torque=env.action_manager.get_term('arm_action').joint_efforts.cpu().numpy(),
                rewards=(env.reward_manager._step_reward*env.step_dt).cpu().numpy(),total=total.cpu().numpy(),
                terms=torch.stack([env.termination_manager.get_term(n) for n in done_names],-1).cpu().numpy(),
                preterminal=(~terminated).cpu().numpy(),action=action.cpu().numpy())
            records.append(row);terminated|=done
            unsafe=row['terms'][:,:3].any()
            if unsafe:
                print('PUSH_ABORT_UNSAFE',phase,'time',row['time_s'],'envs',np.flatnonzero(row['terms'][:,:3].any(-1)).tolist(),flush=True)
                save_trace()
                return
            if args.pace>0:time.sleep(max(0.,env.step_dt*args.pace-(time.perf_counter()-wall)))
        print('PUSH_PHASE',phase,'distance',float((env.scene['object_collection'].data.object_pos_w[:,0,:2]-env.command_manager.get_command('target_goal_pos')[:,:2]).norm(dim=-1).mean()),'terminated',int(terminated.sum()),flush=True)
        save_trace()

cfg=WrenchRefactorEnvCfg();cfg.scene.num_envs=32;cfg.sim.device=args.device;cfg.seed=42
cfg.log_dir=str(args.output);cfg.viewer.eye=(1.2,1.3,2.1);cfg.viewer.lookat=(-.7,0,1.1)
if args.mode=='push':
    cfg.events.object_spawn.func=fixed_push_reset;cfg.events.object_spawn.params={};cfg.curriculum.target_goal_direction_deadzone=None
    cfg.episode_length_s=40.
env=None
try:
    env=gym.make('Inspire-Wrench-Refactor-v0',cfg=cfg).unwrapped;env.reset(seed=42)
    center=env.scene['shelf'].data.root_pos_w[0].cpu().numpy()+np.array([0.,0.,1.1])
    env.sim.set_camera_view((center+np.array([1.2,1.0,.8])).tolist(),center.tolist())
    with torch.inference_mode():
        (reset_audit if args.mode=='reset' else push_audit)(env)
    print('AUDIT_FINISHED',args.mode,flush=True)
    until=time.monotonic()+args.hold_view_seconds
    while app.is_running() and time.monotonic()<until:env.sim.render();time.sleep(.02)
finally:
    if env is not None:env.close()
    app.close()
