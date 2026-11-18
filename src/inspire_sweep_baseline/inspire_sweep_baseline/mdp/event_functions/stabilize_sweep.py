"""Plan a reset without advancing physics, then commit selected environment rows."""
import math
from time import perf_counter
from typing import NamedTuple
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, subtract_frame_transforms, matrix_from_quat
from .wrench_refactor_relative_ik_test import _prepare_fast_ik
from .wrench_refactor import apply_fixed_pool_gravity_mask
from ..common.wrench_refactor_layout import sample_commands
from ..common.stabilize_sweep_reset import hand_pose
from ..common.stabilize_sweep_geometry import RobotResetGeometry, collision_bounds
from ...assets.robot import HAND_CONTACT_BODY_NAMES, HAND_JOINT_NAMES


class InitialPose(NamedTuple):
    joints: torch.Tensor
    boxes: tuple
    position_w: torch.Tensor
    quaternion_w: torch.Tensor
    errors: torch.Tensor
    attempts: torch.Tensor


def _geometry(env):
    if hasattr(env, '_stabilize_robot_geometry'): return
    from isaaclab.sim.utils import get_current_stage
    stage=get_current_stage();robot=env.scene['robot'];path=env.scene.env_prim_paths[0]
    geometry=RobotResetGeometry(stage,path+'/Robot',robot.joint_names,env.device)
    zero=robot.data.default_joint_pos[:1].clone()
    hand_ids=robot.find_joints(list(HAND_JOINT_NAMES),preserve_order=True)[0]
    zero[:,hand_ids]=0.
    pos,quat=env._fast_default_eef
    origin=pos[:1].new_zeros(1,3);identity=quat[:1].new_tensor([[1.,0.,0.,0.]])
    env._stabilize_hand_vertices=geometry.hand_vertices(zero,origin,identity,pos[:1],quat[:1],HAND_CONTACT_BODY_NAMES)
    geometry.exclude_existing_hand_overlaps(zero,origin,identity,HAND_CONTACT_BODY_NAMES)
    env._stabilize_robot_geometry=geometry
    bounds=collision_bounds(stage,path+'/Shelf',path+'/Shelf')
    if not bounds: raise RuntimeError('Shelf has no collision geometry')
    bounds=torch.tensor(bounds,device=env.device,dtype=torch.float32)
    env._stabilize_shelf_boxes=((bounds[:,0]+bounds[:,1])/2,(bounds[:,1]-bounds[:,0])/2)
    env._stabilize_hand_ids=hand_ids


def choose_initial_pose(env, ids, local, delta, shelf_state, root_state):
    _prepare_fast_ik(env);_geometry(env)
    shelf_pos,shelf_quat=shelf_state[:,:3],shelf_state[:,3:7]
    base_pos=root_state[:,:3]+quat_apply(root_state[:,3:7],env._fast_base_offset[0][ids])
    base_quat=quat_mul(root_state[:,3:7],env._fast_base_offset[1][ids])
    default_pos,default_quat=(v[ids] for v in env._fast_default_eef)
    default_world=base_pos+quat_apply(base_quat,default_pos)
    default_local=quat_apply_inverse(shelf_quat,default_world-shelf_pos)
    _,default_shelf_quat=subtract_frame_transforms(shelf_pos,shelf_quat,default_world,quat_mul(base_quat,default_quat))
    bounds=env._revision_bounds_t[ids,0]
    half=(bounds[:,1]-bounds[:,0])/2
    target_quat=local.new_zeros(len(ids),4);target_quat[:,0]=math.sqrt(.5);target_quat[:,3]=-math.sqrt(.5)
    center=local[:,0]+quat_apply(target_quat,bounds.mean(1))
    direction=torch.cat((delta,delta.new_zeros(len(ids),1)),-1);direction/=direction.norm(dim=-1,keepdim=True)
    # Cylinder support is its radius, not the diagonal of its bounding square.
    radius=half[:,:2].amax(-1)
    shelf_centers,shelf_half=env._stabilize_shelf_boxes
    shelf=(shelf_centers.expand(len(ids),-1,-1),shelf_half.expand(len(ids),-1,-1),
           torch.eye(3,device=env.device).expand(len(ids),len(shelf_centers),3,3))
    target=(center[:,None],half[:,None],matrix_from_quat(target_quat)[:,None])
    base_local,base_local_quat=subtract_frame_transforms(shelf_pos,shelf_quat,base_pos,base_quat)
    offset=env.cfg.scene.ee_frame.target_frames[0].offset
    mount_offset=quat_apply_inverse(local.new_tensor(offset.rot)[None],-local.new_tensor(offset.pos)[None])[0]
    robot=env.scene['robot'];joints=robot.data.default_joint_pos[ids].clone();joints[:,env._stabilize_hand_ids]=0.
    valid=torch.zeros(len(ids),device=env.device,dtype=torch.bool)
    attempts=torch.zeros(len(ids),device=env.device,dtype=torch.long)
    desired_world=local.new_zeros(len(ids),3);desired_quat=local.new_zeros(len(ids),4)
    errors=local.new_full((len(ids),2),float('inf'))
    diagnostics=[]
    # Keep commands fixed. Only failed rows try a different initial hand pose;
    # small wrist-roll alternatives retain the same 5mm bottom clearance.
    for elevation in (0.,2.,0.,2.):
        rows=(~valid).nonzero().flatten()
        if not len(rows):break
        attempts[rows]+=1
        draws=torch.rand((len(rows),3),device=env.device,generator=env._goal_reset_rng)
        p,q=hand_pose(center[rows],direction[rows],radius[rows],default_local[rows],default_shelf_quat[rows],
                      env._stabilize_hand_vertices,draws,base_local[rows],mount_offset,math.radians(elevation),
                      target_half_height=half[rows,2])
        pw=shelf_pos[rows]+quat_apply(shelf_quat[rows],p);qw=quat_mul(shelf_quat[rows],q)
        pb,qb=subtract_frame_transforms(base_pos[rows],base_quat[rows],pw,qw)
        angles,ok,pe,re=env._fast_ik.solve(pb,qb,env._fast_initial[ids[rows]],env._fast_limits[ids[rows]])
        candidate=joints[rows].clone();candidate[:,env._goal_reset_arm_ids]=angles
        boxes=env._stabilize_robot_geometry.boxes(candidate,base_local[rows],base_local_quat[rows])
        safe=env._stabilize_robot_geometry.valid(boxes,tuple(v[rows] for v in target),tuple(v[rows] for v in shelf))
        diagnostics.append(dict(env_ids=ids[rows].tolist(), ik_valid=ok.tolist(), collision_free=safe.tolist(),
                                hand_bottom_clearance=.005, wrist_elevation_degrees=elevation,
                                desired_position_shelf=p.tolist(), errors=torch.stack((pe,re),-1).tolist()))
        accepted=ok&safe
        chosen=rows[accepted];joints[chosen]=candidate[accepted];valid[chosen]=True
        desired_world[rows]=pw;desired_quat[rows]=qw;errors[rows]=torch.stack((pe,re),-1)
    boxes=env._stabilize_robot_geometry.boxes(joints,base_local,base_local_quat)
    return InitialPose(joints,boxes,desired_world,desired_quat,errors,attempts),valid,diagnostics


def plan_initial_layout(env, ids, local, shelf_state, root_state, command_params, max_layout_resamples=8):
    """Reject infeasible starts before committing; keep successful rows unchanged.

    Reset failure is not an agent transition. Only IK/collision rejection retries;
    exceptions from broken assets, solvers, or observation code still propagate.
    """
    if isinstance(max_layout_resamples,bool) or not isinstance(max_layout_resamples,int) or max_layout_resamples<0:
        raise ValueError('max_layout_resamples must be a nonnegative integer')
    target,goal=sample_commands(env._revision_bounds_t[ids,0],env._revision_layout_rng,**command_params)
    local[:,0,:2]=target
    pose,valid,diagnostics=choose_initial_pose(env,ids,local,goal-target,shelf_state,root_state)
    # Geometry half-extents may be expanded views; merging requires owned rows.
    pose=pose._replace(boxes=tuple(v.clone() for v in pose.boxes))
    resamples=torch.zeros(len(ids),device=ids.device,dtype=torch.long)
    for _ in range(max_layout_resamples):
        rows=(~valid).nonzero().flatten()
        if not len(rows):break
        target[rows],goal[rows]=sample_commands(env._revision_bounds_t[ids[rows],0],
            env._revision_layout_rng,**command_params)
        local[rows,0,:2]=target[rows]
        resamples[rows]+=1
        retry,accepted,diagnostics=choose_initial_pose(env,ids[rows],local[rows],
            goal[rows]-target[rows],shelf_state[rows],root_state[rows])
        for destination,source in zip((pose.joints,*pose.boxes,pose.position_w,pose.quaternion_w,pose.errors),
                                      (retry.joints,*retry.boxes,retry.position_w,retry.quaternion_w,retry.errors)):
            destination[rows]=source
        pose.attempts[rows]+=retry.attempts
        valid[rows]=accepted
    if not bool(valid.all()):
        raise RuntimeError(f'Stabilize reset exhausted {max_layout_resamples+1} layouts (4 poses/layout): '
                           f'env_ids={ids[~valid].tolist()}, last_attempts={diagnostics}')
    return goal-target,pose,resamples


def commit_scene(env, ids, local, yaw, active, delta, shelf_state, root_state, pose):
    robot,shelf,objects=env.scene['robot'],env.scene['shelf'],env.scene['object_collection']
    shelf_pos,shelf_quat=shelf_state[:,:3],shelf_state[:,3:7]
    world=shelf_pos[:,None]+quat_apply(shelf_quat[:,None].expand(-1,17,-1),local)
    q=local.new_zeros(len(ids),17,4);q[...,0]=(yaw/2).cos();q[...,3]=(yaw/2).sin()
    states=torch.cat((world,quat_mul(shelf_quat[:,None].expand(-1,17,-1),q),local.new_zeros(len(ids),17,6)),-1)
    shelf.write_root_state_to_sim(shelf_state,env_ids=ids)
    robot.write_root_state_to_sim(root_state,env_ids=ids)
    robot.write_joint_state_to_sim(pose.joints,torch.zeros_like(pose.joints),env_ids=ids)
    robot.set_joint_position_target(pose.joints,env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(pose.joints),env_ids=ids)
    objects.write_object_link_state_to_sim(states,env_ids=ids)
    env._active_object_mask[ids]=active;apply_fixed_pool_gravity_mask(env,ids)
    env.target_id[ids,0]=0
    env.target_width[ids,0]=env._fast_visibility_radii[ids,0]
    env.sweep_dir[ids]=quat_apply(shelf_quat,torch.cat((delta,delta.new_zeros(len(ids),1)),-1))
    env._final_actual_population[ids]=active[:,1:].sum(-1)
    env._revision_layout_complete[ids]=True;env._revision_spawn_positions[ids]=world
    env.sensor_data_fresh[ids]=False
    env._initial_obstacle_observation[ids]=0.
    _=robot.data.body_link_pos_w
    env.scene['ee_frame'].reset(ids)
    env._goal_reset_desired_pos_w[ids]=pose.position_w
    env._goal_reset_desired_quat_w[ids]=pose.quaternion_w
    env._goal_reset_errors[ids]=pose.errors


def reset_scene(env, env_ids, spawn_radius=.02, edge_clearance=.02,
                min_goal_distance=.06,max_goal_distance=.12,excluded_x_half_angle=math.pi/6,max_layout_resamples=8):
    start=perf_counter()
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if not len(ids):return
    local=torch.zeros((len(ids),17,3),device=env.device)
    local[...,2]=50.+2.*torch.arange(17,device=env.device)
    local[:,0,2]=env._refactor_spawn_z[ids,0]
    yaw=local.new_zeros(len(ids),17);yaw[:,0]=-math.pi/2
    active=torch.zeros((len(ids),17),device=env.device,dtype=torch.bool);active[:,0]=True
    shelf_state=env.scene['shelf'].data.default_root_state[ids].clone()
    root_state=env.scene['robot'].data.default_root_state[ids].clone()
    shelf_state[:,:3]+=env.scene.env_origins[ids];root_state[:,:3]+=env.scene.env_origins[ids]
    command_params=dict(spawn_radius=spawn_radius,edge_clearance=edge_clearance,
        excluded_x_half_angle=excluded_x_half_angle,min_goal_distance=min_goal_distance,max_goal_distance=max_goal_distance)
    delta,pose,resamples=plan_initial_layout(env,ids,local,shelf_state,root_state,command_params,max_layout_resamples)
    from ..common.stabilize_sweep_layout import place_obstacles
    requested=torch.randint(0,7,(len(ids),),device=env.device,generator=env._revision_layout_rng)
    selected=torch.rand((len(ids),16),device=env.device,generator=env._revision_layout_rng).argsort(-1)[:,:6]+1
    place_obstacles(env._fast_layout,ids,local,yaw,active,delta,pose.boxes,selected,requested,env._revision_layout_rng)
    skipped=requested-active[:,1:].sum(-1)
    targets,obstacles,valid,timings=snapshot(env,ids,local,yaw,active,shelf_state)
    commit_scene(env,ids,local,yaw,active,delta,shelf_state,root_state,pose)
    env._surface_target_w[ids]=targets
    env._surface_obstacles_w[ids]=obstacles
    env._surface_valid[ids]=valid
    env._sweep_phase[ids]=False
    actual=active[:,1:].sum(-1)
    env._reset_diagnostics={
        'Reset/requested_obstacles':requested.float().mean(),
        'Reset/actual_obstacles':actual.float().mean(),
        'Reset/placement_skipped':skipped.float().mean(),
        'Reset/empty_fraction':(actual==0).float().mean(),
        'Reset/ik_attempts':pose.attempts.float().mean(),
        'Reset/layout_resamples':resamples.float().mean(),
        'Reset/ik_recovery_fraction':(resamples>0).float().mean(),
        'Reset/observation_ms':sum(t['total_ms'] for t in timings)/len(ids),
        **{f'Reset/observation_{key}':sum(t.get(key,0.) for t in timings)/len(ids)
           for key in ('upload_ms','readback_ms','ray_ms','components_ms')},
        'Reset/observation_fallbacks':sum(t['fallback'] for t in timings)/len(ids),
        'Reset/target_missing_fallbacks':sum(t['target_missing'] for t in timings)/len(ids),
        'Reset/slot_overflow_fallbacks':sum(t['slot_overflow'] for t in timings)/len(ids),
        'Reset/batch_ms':1000*(perf_counter()-start),
    }


def snapshot(env,ids,local,yaw,active,shelf_state):
    from isaaclab.sim.utils import get_current_stage
    from ..common.stabilize_sweep_surface import SurfaceGeometry
    if not hasattr(env,'_surface_geometry'):
        env._surface_geometry=SurfaceGeometry(get_current_stage())
    targets=local.new_zeros(len(ids),3);obstacles=local.new_zeros(len(ids),16,3)
    valid=torch.zeros((len(ids),16),device=env.device,dtype=torch.bool);timings=[]
    local_cpu,yaw_cpu,active_cpu=(v.detach().cpu().numpy() for v in (local,yaw,active))
    names=tuple(env.cfg.scene.object_collection.rigid_objects)
    for row,index in enumerate(ids.tolist()):
        point,others,timing=env._surface_geometry.observe(env.scene.env_prim_paths[index],
            local_cpu[row],yaw_cpu[row],active_cpu[row],names,device=env.device)
        timing.update(fallback=int(point is None or len(others)>16),target_missing=int(point is None),slot_overflow=int(len(others)>16))
        if point is None or len(others)>16:
            active[row,1:]=False;active_cpu[row,1:]=False
            local[row,1:]=0.;local[row,1:,2]=50.+2.*torch.arange(1,17,device=env.device)
            yaw[row,1:]=0.
            point,others,second=env._surface_geometry.observe(env.scene.env_prim_paths[index],
                local_cpu[row],yaw_cpu[row],active_cpu[row],names,device=env.device)
            for key in ('total_ms','build_ms','upload_ms','readback_ms','ray_ms','components_ms','rays'):
                timing[key]=timing.get(key,0.)+second.get(key,0.)
            if point is None:
                raise RuntimeError(f'Target-only visible snapshot missing after bounded fallback: env={index}')
        targets[row]=torch.as_tensor(point,device=env.device)
        obstacles[row,:len(others)]=torch.as_tensor(others,device=env.device)
        valid[row,:len(others)]=True;timings.append(timing)
    q=shelf_state[:,3:7]
    return (shelf_state[:,:3]+quat_apply(q,targets),
            shelf_state[:,None,:3]+quat_apply(q[:,None].expand(-1,16,-1),obstacles),valid,timings)
