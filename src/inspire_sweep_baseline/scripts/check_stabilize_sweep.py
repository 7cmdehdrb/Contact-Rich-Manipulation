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
parser.add_argument('--num-envs',type=int,default=4)
parser.add_argument('--task',choices=['Inspire-Stabilize-Sweep-v0','Inspire-Sweep-Only-v0','Inspire-Stabilize-Sweep-v0-Test','Inspire-Sweep-Only-v0-Test'],default='Inspire-Stabilize-Sweep-v0')
parser.add_argument('--audit-resets',type=int,default=8)
parser.add_argument('--inject-reset-failures',action='store_true')
parser.add_argument('--output',type=Path,default=Path('/tmp/stabilize_sweep_validation.json'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app


def validate():
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab.utils.math import quat_apply,quat_apply_inverse,convert_quat
    from inspire_sweep_baseline.mdp.common.stabilize_sweep_contact import centroid
    cfg=load_cfg_from_registry(args.task,'env_cfg_entry_point')
    sweep_only=args.task.startswith('Inspire-Sweep-Only-v0')
    finger_test=args.task.endswith('-Test')
    if finger_test:
        if args.inject_reset_failures:raise ValueError('Failure injection belongs to the original reset, not Finger reset')
        from inspire_sweep_baseline.mdp.events.sweep_simple import SweepSimpleEventsCfg
        from inspire_sweep_baseline.mdp.rewards.sweep_simple import SweepSimpleFingerRewardsCfg
        from inspire_sweep_baseline.mdp.event_functions.stabilize_sweep import reset_scene_finger_test
        parent_cfg=load_cfg_from_registry(args.task.removesuffix('-Test'),'env_cfg_entry_point')
        parent_dict=parent_cfg.to_dict();child_dict=cfg.to_dict()
        assert cfg.events.object_spawn.func is reset_scene_finger_test
        assert cfg.events.object_spawn.params==SweepSimpleEventsCfg().object_spawn.params
        child_dict['events']['object_spawn']=parent_dict['events']['object_spawn']
        assert parent_cfg.rewards.reaching is None
        assert cfg.rewards.reaching.to_dict()==SweepSimpleFingerRewardsCfg().reaching.to_dict()
        child_dict['rewards']['reaching']=parent_dict['rewards']['reaching']
        assert child_dict==parent_dict, 'Test variant changed more than reset and Reach'
        base_runner=load_cfg_from_registry(args.task.removesuffix('-Test'),'rsl_rl_cfg_entry_point').to_dict()
        runner=load_cfg_from_registry(args.task,'rsl_rl_cfg_entry_point').to_dict()
        assert runner['experiment_name']==base_runner['experiment_name']+'_test'
        runner['experiment_name']=base_runner['experiment_name']
        assert runner==base_runner
    if sweep_only and not finger_test:
        from inspire_sweep_baseline.envs.stabilize_sweep_envs import StabilizeSweepEnvCfg,SweepOnlyEnvCfg
        from inspire_sweep_baseline.runtime.stabilize_sweep import StabilizeSweepEnv,SweepOnlyEnv
        assert issubclass(SweepOnlyEnvCfg,StabilizeSweepEnvCfg) and issubclass(SweepOnlyEnv,StabilizeSweepEnv)
        parent=StabilizeSweepEnvCfg().to_dict();child=cfg.to_dict()
        for name in ('stage','initial_movement','initial_rotation'):
            assert child['rewards'][name] is None and parent['rewards'][name] is not None
            child['rewards'][name]=parent['rewards'][name]
        assert child==parent, 'Unexpected environment configuration difference'
        base_runner=load_cfg_from_registry('Inspire-Stabilize-Sweep-v0','rsl_rl_cfg_entry_point').to_dict()
        runner=load_cfg_from_registry(args.task,'rsl_rl_cfg_entry_point').to_dict()
        assert runner['experiment_name']=='inspire_sweep_only_v0'
        runner['experiment_name']=base_runner['experiment_name']
        assert runner==base_runner, 'Unexpected PPO configuration difference'
    n=args.num_envs
    if n<2:raise ValueError('Integration checks need at least two environments')
    cfg.seed=42;cfg.scene.num_envs=n
    cfg.scene.ee_frame.debug_vis=False;cfg.commands.target_goal_pos.debug_vis=False
    env=gym.make(args.task,cfg=cfg).unwrapped
    result={'task':args.task,'seed':42,'num_envs':n,'reset_audit':[]}
    if finger_test:
        from pxr import Usd,UsdPhysics
        shelf_root=env.scene.stage.GetPrimAtPath(env.scene.env_prim_paths[0]+'/Shelf')
        bodies=[p for p in Usd.PrimRange(shelf_root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        result['shelf_support_physics']={
            'disable_gravity':bool(env.scene['shelf'].root_physx_view.get_disable_gravities()[0,0]),
            'bodies':[{ 'path':str(p.GetPath()),
                        'kinematic_enabled':bool(UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get())}
                      for p in bodies],
            'fixed_joints':[str(p.GetPath()) for p in Usd.PrimRange(shelf_root) if p.IsA(UsdPhysics.FixedJoint)],
        }
        print('SHELF_SUPPORT_PHYSICS',json.dumps(result['shelf_support_physics']),flush=True)
        assert 'reaching' in env.reward_manager.active_terms
        result['test_contract_reset_and_reach_only']=True
    if sweep_only:
        assert not {'stage','initial_movement','initial_rotation'} & set(env.reward_manager.active_terms)
        assert 'squeeze' in env.reward_manager.active_terms
        result['parent_contract_only_three_reward_terms_removed']=True
        result['ppo_only_experiment_name_changed']=True
    from inspire_sweep_baseline.mdp.event_functions import stabilize_sweep as events
    sample_commands=events.sample_commands
    snapshot=events.snapshot
    snapshots=[]
    def capture_snapshot(env,ids,local,yaw,active,shelf_state,*,batched=False):
        before=tuple(value.clone() for value in (ids,local,yaw,active,shelf_state)) if batched else None
        observed=snapshot(env,ids,local,yaw,active,shelf_state,batched=batched)
        if batched:
            snapshots[:]=[(before,active.clone(),observed)]
        return observed
    sampled=[];plans=[]
    plan_layout=events.plan_initial_layout
    prepare_ik,prepare_geometry=events._prepare_fast_ik,events._geometry
    injection={'kind':'ik','remaining':4 if args.inject_reset_failures else 0}
    original={}
    def capture_plan(env,ids,local,shelf,root,params,max_layout_resamples=8):
        delta,pose,resamples=plan_layout(env,ids,local,shelf,root,params,max_layout_resamples)
        plans.append((ids.clone(),local[:,0,:2].clone(),local[:,0,:2]+delta,resamples.clone()))
        return delta,pose,resamples
    def prepare_with_injection(env):
        prepare_ik(env)
        if 'solve' not in original:
            original['solve']=env._fast_ik.solve
            def solve(*values,**kwargs):
                angles,ok,pe,re=original['solve'](*values,**kwargs)
                if injection['kind']=='ik' and injection['remaining']:
                    ok=ok.clone();ok[0]=False;injection['remaining']-=1
                return angles,ok,pe,re
            env._fast_ik.solve=solve
    def geometry_with_injection(env):
        prepare_geometry(env)
        if 'valid' not in original:
            original['valid']=env._stabilize_robot_geometry.valid
            def valid(*values,**kwargs):
                safe=original['valid'](*values,**kwargs)
                if injection['kind']=='collision' and injection['remaining']:
                    safe=safe.clone();safe[0]=False;injection['remaining']-=1
                return safe
            env._stabilize_robot_geometry.valid=valid
    def capture_commands(*values,**kwargs):
        commands=sample_commands(*values,**kwargs)
        sampled.append(tuple(v.clone() for v in commands))
        return commands
    def check_static_local_com(checkpoint):
        robot=env.scene['robot'];objects=env.scene['object_collection']
        # PhysX returns reusable CPU buffers in XYZW order. Clone before any
        # conversion and compare against fresh physical values, not the cache.
        raw_robot=robot.root_physx_view.get_coms().to(env.device).clone()
        raw_objects=objects.root_physx_view.get_coms().to(env.device).clone()
        raw_robot[...,3:7]=convert_quat(raw_robot[...,3:7],to='wxyz')
        raw_objects[...,3:7]=convert_quat(raw_objects[...,3:7],to='wxyz')
        raw_objects=objects.data._reshape_view_to_data(raw_objects)
        torch.testing.assert_close(robot.data.body_com_pose_b,raw_robot,atol=0,rtol=0)
        torch.testing.assert_close(objects.data.object_com_pose_b,raw_objects,atol=0,rtol=0)
        assert robot.data._body_com_pose_b.timestamp==float('inf')
        assert objects.data._object_com_pose_b.timestamp==float('inf')
        # Isaac's object-pose writer does not invalidate object_com_pose_w at
        # the same physics timestamp. That existing reset-time behavior also
        # occurs without this optimization. Check world COM only after actual
        # physics, where control/contact consumers read the refreshed buffers.
        if checkpoint in ('physics_step','physical_contact'):
            expected_robot=robot.data.body_link_pos_w+quat_apply(robot.data.body_link_quat_w,raw_robot[...,:3])
            expected_objects=objects.data.object_link_pos_w+quat_apply(objects.data.object_link_quat_w,raw_objects[...,:3])
            torch.testing.assert_close(robot.data.body_com_pos_w,expected_robot,atol=1e-6,rtol=0)
            torch.testing.assert_close(objects.data.object_com_pos_w,expected_objects,atol=1e-6,rtol=0)
        checks=result.setdefault('static_local_com_cache_checks',{})
        checks[checkpoint]=checks.get(checkpoint,0)+1
    def native_reset_state():
        objects=env.scene['object_collection'].root_physx_view
        robot=env.scene['robot'].root_physx_view
        return {
            'object_pose':objects.get_transforms().clone(),
            'object_velocity':objects.get_velocities().clone(),
            'object_gravity':objects.get_disable_gravities().clone(),
            'object_simulation':objects.get_disable_simulations().clone(),
            'root_pose':robot.get_root_transforms().clone(),
            'root_velocity':robot.get_root_velocities().clone(),
            'joint_position':robot.get_dof_positions().clone(),
            'joint_velocity':robot.get_dof_velocities().clone(),
        }
    def reset_rng_state():
        return {'layout':env._revision_layout_rng.get_state().clone(),
                'goal':env._goal_reset_rng.get_state().clone(),
                'cpu':torch.random.get_rng_state().clone(),
                'cuda':torch.cuda.get_rng_state(env.device).clone()}
    def restore_reset_rng(state):
        env._revision_layout_rng.set_state(state['layout'])
        env._goal_reset_rng.set_state(state['goal'])
        torch.random.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'],env.device)
    def sensor_roster():
        return tuple((name,id(sensor),id(getattr(sensor,'_contact_physx_view',None)))
                     for name,sensor in env.scene.sensors.items())
    def collision_filter_state():
        stage=env.scene.stage
        physics=stage.GetPrimAtPath(env.scene.physics_scene_path)
        assert physics.GetAttribute('physxScene:invertCollisionGroupFilter').Get() is True
        groups=[]
        for row,path in enumerate(env.scene.env_prim_paths):
            group_path=f'/World/collisions/group{row}'
            group=stage.GetPrimAtPath(group_path)
            includes=tuple(map(str,group.GetRelationship('collection:colliders:includes').GetTargets()))
            filtered=tuple(map(str,group.GetRelationship('physics:filteredGroups').GetTargets()))
            assert includes==(path,) and filtered==(group_path,)
            groups.append((includes,filtered))
        return tuple(groups)
    filter_before=collision_filter_state()
    def check_test_parking(*, contacts=False):
        from inspire_sweep_baseline.mdp.common.stabilize_sweep_layout import test_inactive_positions
        objects=env.scene['object_collection']
        shelf=env.scene['shelf'].data.default_root_state
        # Parking is fixed at reset, independent of subsequent dynamic shelf motion.
        expected=(shelf[:,:3]+env.scene.env_origins)[:,None]+quat_apply(
            shelf[:,None,3:7].expand(-1,17,-1),test_inactive_positions(env))
        actual=objects.root_physx_view.get_transforms().reshape(17,n,7).transpose(0,1)
        inactive=~env._active_object_mask
        assert not inactive[:,0].any()
        torch.testing.assert_close(actual[...,:3][inactive],expected[inactive],atol=1e-6,rtol=0)
        rows=inactive.T.flatten()
        assert not objects.root_physx_view.get_velocities()[rows].any()
        assert objects.root_physx_view.get_disable_gravities()[rows.cpu()].all()
        assert not objects.root_physx_view.get_disable_simulations().any()
        assert collision_filter_state()==filter_before
        result['test_inactive_parking_grid_and_unchanged_collision_groups']=True
        if contacts:
            from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
            for source in range(1,17):
                forces=env.scene[object_hand_sensor_name(objects.object_names[source])].data.net_forces_w
                assert not forces[inactive[:,source]].any(), f'Parked body contacted another body: {source}'
            result['test_inactive_bodies_stationary_and_contact_free_after_physics']=True
    def check_commands():
        if sweep_only:
            state=env._stabilize_state
            assert state.achieved.all() and state.reward_phase.all() and not state.consecutive.any()
            assert (env.observation_manager.compute()['policy'][:,-1]==1.).all()
        shelf=env.scene['shelf'].data
        target=env.scene['object_collection'].data.object_link_pos_w[:,0]
        goal=env.command_manager.get_command('target_goal_pos')[:,:3]
        check_static_local_com('reset')
        check_test_parking()
        # Replay the same *pre-fallback* scene through the reference so an
        # unnecessary target-only fallback cannot hide an observer mismatch.
        before,actual_active,observed=snapshots[-1]
        ids,local,yaw,active,shelf_state=(value.clone() for value in before)
        reference_target,reference_obstacles,reference_valid,reference_timings=snapshot(
            env,ids,local,yaw,active,shelf_state)
        torch.testing.assert_close(active,actual_active)
        torch.testing.assert_close(reference_valid,observed[2])
        torch.testing.assert_close(reference_target,observed[0],atol=2e-5,rtol=0)
        torch.testing.assert_close(reference_obstacles,observed[1],atol=2e-5,rtol=0)
        for expected,actual in zip(reference_timings,observed[3]):
            for key in ('fallback','target_missing','slot_overflow'):
                assert expected[key]==actual[key]
        result['batched_surface_matches_original_mesh_observer']=True
        if finger_test:
            assert not hasattr(env,'_stabilize_robot_geometry')
            assert (env._goal_reset_errors[:,0]<=.0005).all() and (env._goal_reset_errors[:,1]<=.003).all()
            desired=quat_apply_inverse(shelf.root_quat_w,env._goal_reset_desired_pos_w-shelf.root_pos_w)
            local_target=quat_apply_inverse(shelf.root_quat_w,target-shelf.root_pos_w)
            assert ((desired[:,0]>=.28-1e-5)&(desired[:,0]<=.30+1e-5)).all()
            lateral=(desired[:,1]-local_target[:,1]).abs()
            assert ((lateral>=.10-1e-5)&(lateral<=.18+1e-5)).all()
            assert ((env._surface_target_w-centroid(env)).norm(dim=-1)>.01).all()
            result['finger_initial_pose_and_surface_observation']=True
            return
        torch.testing.assert_close(quat_apply_inverse(shelf.root_quat_w,target-shelf.root_pos_w)[:,:2],plans[-1][1],atol=2e-5,rtol=0)
        torch.testing.assert_close(quat_apply_inverse(shelf.root_quat_w,goal-shelf.root_pos_w)[:,:2],plans[-1][2],atol=2e-5,rtol=0)
        assert (env._goal_reset_errors[:,0]<=.0005).all()
        assert (env._goal_reset_errors[:,1]<=.003).all()
        # Read the real right-hand tactile body, independently of the helper's EEF axis.
        robot=env.scene['robot']
        palm_id=robot.find_bodies('inspire_palm_force_sensor')[0][0]
        palm_normal=quat_apply(robot.data.body_link_quat_w[:,palm_id],
                               target.new_tensor([0.,0.,1.]).expand(n,-1))
        direction=goal-target;direction[:,2]=0.
        direction/=direction.norm(dim=-1,keepdim=True)
        direction_shelf=quat_apply_inverse(shelf.root_quat_w,direction)
        side=torch.where(direction_shelf[:,1]>0,-1.,1.)
        alignment=(palm_normal*direction).sum(-1)*side
        assert (alignment>0.98).all(), f'Palm/dorsum misaligned with goal: {alignment.tolist()}'
        palm_shelf=quat_apply_inverse(shelf.root_quat_w,palm_normal)
        assert (palm_shelf[:,1]<0).all(), 'Right palm must remain on the robot-left side'
        result['minimum_actual_contact_face_goal_cosine']=min(
            result.get('minimum_actual_contact_face_goal_cosine',1.),float(alignment.min()))
        frame=env.scene['ee_frame'].data
        vertices=env._stabilize_hand_vertices.expand(n,-1,-1)
        surface=frame.target_pos_w[:,0,None]+quat_apply(
            frame.target_quat_w[:,0,None].expand(-1,vertices.shape[1],-1),vertices)
        surface=quat_apply_inverse(shelf.root_quat_w[:,None].expand(-1,vertices.shape[1],-1),
                                   surface-shelf.root_pos_w[:,None])
        center_shelf=quat_apply_inverse(shelf.root_quat_w,centroid(env)-shelf.root_pos_w)
        half_height=(env._revision_bounds_t[:,0,1,2]-env._revision_bounds_t[:,0,0,2])/2
        bottom=surface[...,2].amin(-1)
        clearance=bottom-(center_shelf[:,2]-half_height)
        torch.testing.assert_close(clearance,torch.full_like(clearance,.005),atol=.001,rtol=0)
        assert (bottom<center_shelf[:,2]+half_height).all(), 'Target fits entirely under the hand'
        for key,value,reduce in [('minimum_hand_bottom_clearance_m',float(clearance.min()),min),
                                 ('maximum_hand_bottom_clearance_m',float(clearance.max()),max)]:
            result[key]=reduce(result.get(key,value),value)
        result['palm_always_left']=True
    events.sample_commands=capture_commands
    events.snapshot=capture_snapshot
    events.plan_initial_layout=capture_plan
    if args.inject_reset_failures:
        events._prepare_fast_ik=prepare_with_injection
        events._geometry=geometry_with_injection
    try:
        # Match train.py's first reset, including the reported four-env seed-42 failure.
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        wrapper=RslRlVecEnvWrapper(env,clip_actions=1.5707963267948966)
        assert torch.isfinite(wrapper.get_observations()['policy']).all()
        if args.inject_reset_failures:
            assert injection['remaining']==0 and len(sampled)>=2
            assert plans[-1][3][0]>=1 and not plans[-1][3][1:].any()
            torch.testing.assert_close(plans[-1][1][1:],sampled[0][0][1:])
            torch.testing.assert_close(plans[-1][2][1:],sampled[0][1][1:])
            assert not env.episode_length_buf.any() and not env.termination_manager.dones.any()
            assert not env._timeout_outcomes.totals.any()
            result['injected_initial_ik_failure_recovered']=True
        elif not finger_test:
            assert sum(len(v[0]) for v in sampled)==n+int(plans[-1][3].sum())
        check_commands()
        result['training_wrapper_initial_reset']=True
        # Audit ordinary resets separately from injected failures.
        for seed in range(args.audit_resets):
            start=perf_counter();before=len(sampled)
            try:
                obs,_=env.reset(seed=seed)
                torch.cuda.synchronize()
                reset_ms=1000*(perf_counter()-start)
                if not finger_test:assert sum(len(v[0]) for v in sampled[before:])==n+int(plans[-1][3].sum())
                check_commands()
                result['reset_audit'].append(dict(seed=seed,ok=True,ms=reset_ms,
                    max_position_error_m=float(env._goal_reset_errors[:,0].max()),
                    max_rotation_error_rad=float(env._goal_reset_errors[:,1].max()),
                    metrics={k:float(v) for k,v in env.extras['log'].items() if k.startswith('Reset/')}))
            except RuntimeError as error:
                if 'layouts (4 poses/layout)' not in str(error):raise
                result['reset_audit'].append(dict(seed=seed,ok=False,error=str(error),ms=1000*(perf_counter()-start)))
        assert all(row['ok'] for row in result['reset_audit']), 'Initial pose failed in reset audit'
        if not finger_test:result['bounded_failed_row_resampling']=True
        else:
            # Differential check against the exact Finger reset on identical assets/RNG.
            from inspire_sweep_baseline.mdp.event_functions.wrench_refactor_relative_ik_test import fast_reset_scene
            env.reset(seed=123)
            expected_joints=env.scene['robot'].data.joint_pos.clone()
            expected_objects=env.scene['object_collection'].data.object_link_state_w.clone()
            expected_active=env._active_object_mask.clone()
            expected_goal=env.command_manager.get_command('target_goal_pos')[:,:3].clone()
            env._revision_layout_rng.manual_seed(123);env._goal_reset_rng.manual_seed(123+1707)
            before_rng=reset_rng_state();before_roster=sensor_roster()
            layout=env._fast_layout
            original_batch_size=layout.sample_batch_size
            # Remove Test instance accelerators as well as restoring batch64:
            # the reference must execute Finger's original Torch sampler.
            overrides={name:layout.__dict__.pop(name) for name in ('red_distance','cell_selector')
                       if name in layout.__dict__}
            layout.sample_batch_size=64
            try:
                fast_reset_scene(env,torch.arange(n,device=env.device),**cfg.events.object_spawn.params)
                reference_native=native_reset_state();reference_rng=reset_rng_state()
            finally:
                layout.sample_batch_size=original_batch_size
                layout.__dict__.update(overrides)
            result['reference_finger_sampler_without_test_overrides']=True
            torch.testing.assert_close(env.scene['robot'].data.joint_pos,expected_joints,atol=1e-6,rtol=0)
            torch.testing.assert_close(env.scene['object_collection'].data.object_link_state_w[expected_active],
                                       expected_objects[expected_active],atol=1e-5,rtol=0)
            assert not (expected_active&~env._active_object_mask).any()
            target_root=env.scene['object_collection'].data.object_link_pos_w[:,0]
            torch.testing.assert_close(target_root+env.sweep_dir,expected_goal,atol=1e-5,rtol=0)
            result['same_seed_matches_original_finger_initialization']=True
            restore_reset_rng(before_rng)
            from inspire_sweep_baseline.mdp.common.stabilize_sweep_layout import (
                configure_test_layout_batch,test_inactive_positions)
            configure_test_layout_batch(env)
            fast_reset_scene(env,torch.arange(n,device=env.device),**cfg.events.object_spawn.params,
                             reuse_unchanged_states=True,inactive_positions=test_inactive_positions(env),
                             populate_legacy_observation=False)
            assert not env._initial_obstacle_observation.any()
            result['test_legacy_centroid_buffer_unused_and_zero']=True
            for key,value in native_reset_state().items():
                expected=reference_native[key]
                if key=='object_pose':
                    # Only inactive XYZ deliberately changes; even parked
                    # orientations, velocities and native flags stay exact.
                    torch.testing.assert_close(value[:,3:],expected[:,3:],atol=0,rtol=0,msg=key+' quaternion')
                    active_rows=env._active_object_mask.T.flatten()
                    value,expected=value[active_rows],expected[active_rows]
                torch.testing.assert_close(value,expected,atol=0,rtol=0,msg=key)
            for key,value in reset_rng_state().items():
                torch.testing.assert_close(value,reference_rng[key],atol=0,rtol=0,msg=key)
            assert sensor_roster()==before_roster
            check_test_parking()
            result['test_reset_matches_finger_except_inactive_xyz_and_all_rng']=True
            result['reset_native_equality']='numerical exact: atol=rtol=0, +0/-0 equivalent'
        # Separate selected-row resets use an explicit reproducible fixture seed.
        for row in range(n):obs,_=env.reset(seed=42,env_ids=torch.tensor([row],device=env.device))
        assert obs['policy'].shape==(n,67) and obs['obstacles'].shape==(n,64)
        target=env._surface_target_w.clone();points=env._surface_obstacles_w.clone()
        center=centroid(env);data=env.scene['object_collection'].data
        offset=env._revision_bounds_t[:,0].mean(1)
        torch.testing.assert_close(center,data.object_link_pos_w[:,0]+quat_apply(data.object_link_quat_w[:,0],offset))
        result['centroid_com']={'height':(env._revision_bounds_t[:,0,1,2]-env._revision_bounds_t[:,0,0,2]).tolist(),
            'centroid_minus_root':(center-data.object_link_pos_w[:,0]).tolist(),
            'com_minus_root':(data.object_com_pos_w[:,0]-data.object_link_pos_w[:,0]).tolist()}
        held=env.scene['object_collection'].data.object_link_state_w[1].clone()
        joints=env.scene['robot'].data.joint_pos[1].clone();clock=env._sim_step_counter
        if finger_test:
            held_native=native_reset_state();held_roster=sensor_roster()
            held_legacy=env._initial_obstacle_observation.clone()
            env._initial_obstacle_observation[0]=torch.nan
        if finger_test:
            reach_cfg=env.reward_manager.get_term_cfg('reaching')
            reach=reach_cfg.func
            held_reach=reach._initial_distance[1].clone()
        if args.inject_reset_failures:injection.update(kind='collision',remaining=4)
        env.reset(seed=42,env_ids=torch.tensor([0],device=env.device))
        if args.inject_reset_failures:
            assert injection['remaining']==0 and plans[-1][3][0]>=1
            result['injected_partial_collision_failure_recovered']=True
            target[0]=env._surface_target_w[0];points[0]=env._surface_obstacles_w[0]
        assert env._sim_step_counter==clock
        torch.testing.assert_close(held,env.scene['object_collection'].data.object_link_state_w[1],atol=0,rtol=0)
        torch.testing.assert_close(joints,env.scene['robot'].data.joint_pos[1],atol=0,rtol=0)
        torch.testing.assert_close(target[1],env._surface_target_w[1],atol=0,rtol=0)
        if finger_test:
            # Inspect every untouched native row, including remote pooled
            # bodies and physical flags, independently of lazy data caches.
            untouched_objects=(torch.arange(17)[:,None]*n+torch.arange(1,n)[None]).flatten()
            for key,value in native_reset_state().items():
                rows=untouched_objects if key.startswith('object_') else torch.arange(1,n)
                rows=rows.to(value.device)
                torch.testing.assert_close(value[rows],held_native[key][rows],atol=0,rtol=0,msg=key)
            assert sensor_roster()==held_roster
            assert not env._initial_obstacle_observation[0].any()
            torch.testing.assert_close(env._initial_obstacle_observation[1:],held_legacy[1:],atol=0,rtol=0)
            result['partial_reset_clears_only_selected_legacy_rows']=True
            result['partial_reset_native_state_flags_and_roster_isolation']=True
            torch.testing.assert_close(reach._initial_distance[1],held_reach,atol=0,rtol=0)
            expected_initial=((env.scene['ee_frame'].data.target_pos_w[0,0]
                               -env.scene['object_collection'].data.object_pos_w[0,env.target_id[0,0].long()]).norm()-.1).clamp_min(0.)
            torch.testing.assert_close(reach._initial_distance[0],expected_initial)
            check_static_local_com('partial_reset')
        result['partial_reset_isolation']=True
        start=perf_counter()
        for _ in range(4):
            obs,reward,terminated,truncated,_=env.step(torch.zeros((n,12),device=env.device))
            assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
            assert not (terminated|truncated).any()
            if sweep_only:assert (obs['policy'][:,-1]==1.).all() and env._stabilize_state.reward_phase.all()
            if finger_test:
                distance=(env.scene['ee_frame'].data.target_pos_w[:,0]-centroid(env)).norm(dim=-1)
                expected_reach=reach._initial_distance/env.cfg.episode_length_s*.1/(.1+(distance-.1).clamp_min(0.))
                torch.testing.assert_close(reach(env,**reach_cfg.params),expected_reach)
                index=env.reward_manager.active_terms.index('reaching')
                torch.testing.assert_close(env.reward_manager._step_reward[:,index],expected_reach*30.)
                assert (expected_reach>0).all()
                result['reach_matches_finger_formula_and_partial_reset']=True
        torch.cuda.synchronize();result['real_step_ms']=1000*(perf_counter()-start)/4
        check_static_local_com('physics_step')
        check_test_parking(contacts=True)
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
                   torch.tensor([[.5,0.,0.]],device=env.device).expand(n,-1),
                   torch.zeros((n,3),device=env.device),torch.full((n,),.5,device=env.device),
                   torch.ones(n,device=env.device,dtype=torch.bool),torch.zeros(n,device=env.device),torch.zeros(n,device=env.device))
        state.sample=replay
        try:
            for step in range(15):
                obs,reward,terminated,truncated,_=env.step(torch.zeros((n,12),device=env.device))
                assert not (terminated|truncated).any()
                assert bool((obs['policy'][:,-1]==(1. if sweep_only or step==14 else 0.)).all())
            if sweep_only:
                assert state.reward_phase.all() and state.achieved.all() and not state.consecutive.any()
                assert not state.setup_sum.any() and not state.move_cost.any() and not state.rotate_cost.any()
            else:assert not state.reward_phase.any() and state.achieved.all()
            obs,*_=env.step(torch.zeros((n,12),device=env.device))
            assert state.reward_phase.all() and (state.stage_score==1.).all()
            result['sweep_from_reset_without_gate' if sweep_only else 'controlled_gate_manager_boundary']=True
            # Accelerated terminal counter: same-step Gate latch counts at timeout,
            # and only that row resets while the other stays in Sweep.
            if not sweep_only:state.achieved[0]=False;state.consecutive[0]=28
            env.episode_length_buf[0]=env.max_episode_length-1
            env.command_manager.get_term('target_goal_pos').pos_command_w[0,:2]=centroid(env)[0,:2]+.005
            env._revision_layout_rng.manual_seed(42);env._goal_reset_rng.manual_seed(42+1707)
            before=int(env._timeout_outcomes.totals[0])
            if args.inject_reset_failures:injection.update(kind='ik',remaining=4)
            obs,reward,terminated,truncated,_=env.step(torch.zeros((n,12),device=env.device))
            assert truncated.tolist()==[True]+[False]*(n-1) and not terminated.any()
            assert int(env._timeout_outcomes.totals[0])==before+1
            assert obs['policy'][:,-1].tolist()==([1.]*n if sweep_only else [0.]+[1.]*(n-1))
            result['same_step_gate_timeout_and_partial_reset']=True
            check_static_local_com('timeout_partial_reset')
            if args.inject_reset_failures:
                assert injection['remaining']==0 and plans[-1][3][0]>=1
                assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
                assert env.episode_length_buf[0]==0
                result['injected_auto_reset_ik_failure_recovered']=True
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
            normal_sum=torch.zeros(n,3);total=torch.zeros(n,3);torque=torch.zeros(n,3);load=torch.zeros(n)
            for row in range(n):
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
            check_static_local_com('physical_contact')
            check_test_parking(contacts=True)
            measured=True;break
        assert measured,'No physical hand/target contact found in the controlled probe'
        result['final_reset_metrics']={k:float(v) for k,v in env.extras['log'].items() if k.startswith('Reset/')}
    finally:
        events.sample_commands=sample_commands
        events.snapshot=snapshot
        events.plan_initial_layout=plan_layout
        events._prepare_fast_ik=prepare_ik;events._geometry=prepare_geometry
        if 'solve' in original:env._fast_ik.solve=original['solve']
        if 'valid' in original:env._stabilize_robot_geometry.valid=original['valid']
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
