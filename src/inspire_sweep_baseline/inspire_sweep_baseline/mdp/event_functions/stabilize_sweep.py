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
    write_test_reset_root_state(env,ids,root_state)
    robot.write_joint_state_to_sim(pose.joints,torch.zeros_like(pose.joints),env_ids=ids)
    robot.set_joint_position_target(pose.joints,env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(pose.joints),env_ids=ids)
    written=write_test_reset_object_state(env,ids,states,active)
    env._active_object_mask[ids]=active
    # Same fixed-pool gravity/wake contract as the optimized Finger adapter.
    rows=(torch.arange(17)[:,None]*env.num_envs+ids.cpu()[None]).flatten()
    flags=(~active).T.reshape(-1,1).to(device='cpu',dtype=torch.uint8)
    changed=(env._wrench_gravity_flags[rows]!=flags).flatten()
    env._wrench_gravity_flags[rows]=flags
    if bool(changed.any()):
        objects.root_physx_view.set_disable_gravities(env._wrench_gravity_flags,rows[changed])
    wake_rows=rows[written.T.flatten().cpu()]
    if len(wake_rows):objects.root_physx_view.wake_up(wake_rows)
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
    from ..common.stabilize_sweep_contact import cache_static_local_com
    from ..common.stabilize_sweep_layout import test_inactive_positions, fused_test_red_distance
    parking=test_inactive_positions(env)[ids]
    env._fast_layout.red_distance=fused_test_red_distance
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
    place_obstacles(env._fast_layout,ids,local,yaw,active,delta,pose.boxes,selected,requested,env._revision_layout_rng,
                    batch_size=32)
    skipped=requested-active[:,1:].sum(-1)
    targets,obstacles,valid,timings=snapshot(env,ids,local,yaw,active,shelf_state,batched=True)
    # Planning and visibility retain their original coordinates. Only retired
    # bodies change physical parking, including target-only fallback rows.
    local=torch.where(active[...,None],local,parking)
    commit_scene(env,ids,local,yaw,active,delta,shelf_state,root_state,pose)
    cache_static_local_com(env)
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


def snapshot(env,ids,local,yaw,active,shelf_state,*,batched=False):
    if batched:
        return _snapshot_batched(env,ids,local,yaw,active,shelf_state)
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


def _snapshot_batched(env,ids,local,yaw,active,shelf_state):
    """Shared IK/Test observer: bulk transfers with the same snapshot contract."""
    import numpy as np
    from ..common.stabilize_sweep_surface import BatchedSurfaceGeometry
    if not hasattr(env,'_surface_batch_geometry'):
        env._surface_batch_geometry=BatchedSurfaceGeometry(
            env,chunk_size=256,components_backend='gpu',ray_backend='raster',bounds_backend='cpu_kernel')
    geometry=env._surface_batch_geometry
    ids_cpu=ids.detach().cpu().numpy()
    local_cpu,yaw_cpu,active_cpu=(v.detach().cpu().numpy().copy() for v in (local,yaw,active))
    results=geometry.observe_batch(ids_cpu,local_cpu,yaw_cpu,active_cpu)
    failed=[]
    timings=[]
    for row,(point,others,timing) in enumerate(results):
        timing.update(fallback=int(point is None or len(others)>16),
                      target_missing=int(point is None),slot_overflow=int(len(others)>16))
        timings.append(timing)
        if timing['fallback']:failed.append(row)
    if failed:
        active_cpu[failed,1:]=False
        local_cpu[failed,1:]=0.
        local_cpu[np.asarray(failed)[:,None],np.arange(1,17),2]=50.+2.*np.arange(1,17)
        yaw_cpu[failed,1:]=0.
        retry=geometry.observe_batch(ids_cpu[failed],local_cpu[failed],yaw_cpu[failed],active_cpu[failed])
        for row,(point,others,second) in zip(failed,retry):
            for key in ('total_ms','build_ms','upload_ms','readback_ms','ray_ms','components_ms','rays'):
                timings[row][key]=timings[row].get(key,0.)+second.get(key,0.)
            if point is None:
                raise RuntimeError(f'Target-only visible snapshot missing after bounded fallback: env={ids_cpu[row]}')
            results[row]=(point,others,timings[row])
        # Preserve the original fallback's physical state changes in selected rows.
        rows=torch.as_tensor(failed,device=env.device)
        active[rows]=torch.as_tensor(active_cpu[failed],device=env.device)
        local[rows]=torch.as_tensor(local_cpu[failed],device=env.device)
        yaw[rows]=torch.as_tensor(yaw_cpu[failed],device=env.device)
    target_cpu=np.empty((len(ids),3),dtype=np.float32)
    obstacles_cpu=np.zeros((len(ids),16,3),dtype=np.float32)
    valid_cpu=np.zeros((len(ids),16),dtype=bool)
    for row,(point,others,_) in enumerate(results):
        target_cpu[row]=point
        obstacles_cpu[row,:len(others)]=others
        valid_cpu[row,:len(others)]=True
    targets=torch.as_tensor(target_cpu,device=env.device)
    obstacles=torch.as_tensor(obstacles_cpu,device=env.device)
    valid=torch.as_tensor(valid_cpu,device=env.device)
    q=shelf_state[:,3:7]
    return (shelf_state[:,:3]+quat_apply(q,targets),
            shelf_state[:,None,:3]+quat_apply(q[:,None].expand(-1,16,-1),obstacles),valid,timings)


def write_test_reset_root_state(env, ids, state):
    """Skip identical fixed-base native root writes; preserve Isaac reset caches.

    Native getters, rather than same-timestamp data caches, decide equality.
    Joint writes still run normally afterwards and invalidate link pose caches.
    Shared by the fixed-physics Stabilize/Sweep Only IK and Test lifecycles.
    """
    robot = env.scene['robot']
    if not robot.is_fixed_base:
        robot.write_root_state_to_sim(state, env_ids=ids)
        return
    from isaaclab.utils.math import convert_quat
    pose = robot.root_physx_view.get_root_transforms()[ids].clone()
    pose[:, 3:] = convert_quat(pose[:, 3:], to='wxyz')
    velocity = robot.root_physx_view.get_root_velocities()[ids]
    changed = (~env._revision_layout_complete[ids] | (pose != state[:, :7]).any(-1)
               | (velocity != state[:, 7:]).any(-1))
    data = robot.data
    data.root_link_pose_w[ids] = state[:, :7].clone()
    for name in ('_root_link_state_w', '_root_state_w'):
        if getattr(data, name).data is not None:
            getattr(data, name[1:])[ids, :7] = data.root_link_pose_w[ids]
    poses_xyzw = data.root_link_pose_w.clone()
    poses_xyzw[:, 3:] = convert_quat(poses_xyzw[:, 3:], to='xyzw')
    for name in ('_body_link_pose_w', '_body_com_pose_w', '_body_state_w',
                 '_body_link_state_w', '_body_com_state_w'):
        getattr(data, name).timestamp = -1.
    if bool(changed.any()):
        robot.root_physx_view.set_root_transforms(poses_xyzw, indices=ids[changed])
    data.root_com_vel_w[ids] = state[:, 7:].clone()
    for name in ('_root_com_state_w', '_root_state_w'):
        if getattr(data, name).data is not None:
            getattr(data, name[1:])[ids, 7:] = data.root_com_vel_w[ids]
    # The original root velocity writer clears every selected body's cached
    # acceleration even when its native root velocity was already zero.
    data.body_acc_w[ids] = 0.
    if bool(changed.any()):
        robot.root_physx_view.set_root_velocities(data.root_com_vel_w, indices=ids[changed])


def write_test_reset_object_state(env, ids, state, active):
    """Write only changed native rows; preserve the installed Isaac cache contract.

    The PhysX tensor view uses object-major indices and full-sized setter inputs.
    Only previously parked, still inactive bodies with exactly equal native pose and
    zero velocity can skip writes and wake-up. First reset, active bodies, and
    both activity transitions always use the native setters. No gravity,
    collision, simulation-enable, or sleep flags are changed here.

    Cache updates mirror RigidObjectCollection's link-state writer in the
    installed Isaac Lab version, including its same-timestamp world-COM behavior.
    Revalidate this helper if that upstream writer or Test physics changes.
    """
    from isaaclab.utils.math import combine_frame_transforms, convert_quat
    objects = env.scene['object_collection']
    data = objects.data
    native_pose = objects.root_physx_view.get_transforms().clone()
    native_velocity = objects.root_physx_view.get_velocities()
    pose = objects.reshape_view_to_data(native_pose)[ids].clone()
    pose[..., 3:] = convert_quat(pose[..., 3:], to='wxyz')
    velocity = objects.reshape_view_to_data(native_velocity)[ids]
    written = (active | env._active_object_mask[ids]
               | ~env._revision_layout_complete[ids, None]
               | (pose != state[..., :7]).any(-1) | (velocity != state[..., 7:]).any(-1))
    # This reset only requests zero link velocity, which is also zero COM
    # velocity. Reject future nonzero callers instead of silently skipping the
    # link-to-COM velocity transform performed by Isaac's general writer.
    torch._assert_async((state[..., 7:] == 0.).all(), 'Test reset requires zero object velocity')
    object_ids = objects._ALL_OBJ_INDICES
    selected = (ids[:, None], object_ids)
    data.object_link_pose_w[selected] = state[..., :7].clone()
    if data._object_link_state_w.data is not None:
        data.object_link_state_w[ids[:, None], object_ids, :7] = state[..., :7].clone()
    if data._object_state_w.data is not None:
        data.object_state_w[ids[:, None], object_ids, :7] = state[..., :7].clone()
    if data._object_com_state_w.data is not None:
        pos, quat = combine_frame_transforms(state[..., :3], state[..., 3:7],
                                             data.object_com_pos_b[selected], data.object_com_quat_b[selected])
        data.object_com_state_w[ids[:, None], object_ids, :3] = pos
        data.object_com_state_w[ids[:, None], object_ids, 3:7] = quat
    poses_xyzw = data.object_link_pose_w.clone()
    poses_xyzw[..., 3:] = convert_quat(poses_xyzw[..., 3:], to='xyzw')
    rows = (object_ids[:, None] * env.num_envs + ids[None]).flatten()[written.T.flatten()]
    if len(rows):
        objects.root_physx_view.set_transforms(objects.reshape_data_to_view(poses_xyzw), indices=rows)
    data.object_link_vel_w[selected] = state[..., 7:].clone()
    if data._object_link_state_w.data is not None:
        data.object_link_state_w[ids[:, None], object_ids, 7:] = state[..., 7:].clone()
    data.object_com_vel_w[selected] = state[..., 7:].clone()
    for name in ('_object_com_state_w', '_object_state_w', '_object_link_state_w'):
        if getattr(data, name).data is not None:
            getattr(data, name[1:])[ids[:, None], object_ids, 7:] = state[..., 7:].clone()
    data.object_com_acc_w[selected] = 0.
    if len(rows):
        objects.root_physx_view.set_velocities(objects.reshape_data_to_view(data.object_com_vel_w), indices=rows)
    return written


def reset_scene_finger_test(env, env_ids, ik_params, spawn_radius=.02, edge_clearance=.02,
                            excluded_x_half_angle=math.pi/6, min_goal_distance=.06,
                            max_goal_distance=.12, observation_noise=.01,
                            visibility_threshold=.35, visibility_flip_probability=.01,
                            observation_swap_probability=.1):
    """Finger's active circle/IK reset with nearby inactive parking and surface points.

    The legacy centroid observation produced by fast_reset_scene is not exposed
    to this policy. No robot hull construction or Stabilize pose search is used.
    """
    from .wrench_refactor_relative_ik_test import fast_reset_scene
    from ..common.stabilize_sweep_contact import cache_static_local_com
    from ..common.stabilize_sweep_layout import configure_test_layout_batch, test_inactive_positions
    from isaaclab.utils.math import quat_conjugate
    start=perf_counter()
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if not len(ids):return
    configure_test_layout_batch(env)
    parking = test_inactive_positions(env)[ids]
    fast_reset_scene(env,ids,ik_params=ik_params,spawn_radius=spawn_radius,edge_clearance=edge_clearance,
        excluded_x_half_angle=excluded_x_half_angle,min_goal_distance=min_goal_distance,max_goal_distance=max_goal_distance,
        observation_noise=observation_noise,visibility_threshold=visibility_threshold,
        visibility_flip_probability=visibility_flip_probability,observation_swap_probability=observation_swap_probability,
        reuse_unchanged_states=True, inactive_positions=parking, populate_legacy_observation=False)
    cache_static_local_com(env)
    placement_ms=1000*(perf_counter()-start)
    objects=env.scene['object_collection']
    shelf_state=env.scene['shelf'].data.root_state_w[ids]
    states=objects.data.object_link_state_w[ids].clone()
    q=shelf_state[:,None,3:7].expand(-1,17,-1)
    local=quat_apply_inverse(q,states[:,:,:3]-shelf_state[:,None,:3])
    relative=quat_mul(quat_conjugate(q),states[:,:,3:7])
    w,x,y,z=relative.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y.square()+z.square()))
    active=env._active_object_mask[ids].clone()
    requested=active[:,1:].sum(-1)
    targets,obstacles,valid,timings=snapshot(env,ids,local,yaw,active,shelf_state,batched=True)
    # The target-only fallback retires obstacles before its existing commit.
    # Keep those bodies on the same nearby grid, with no far-parking teleport.
    local = torch.where(active[..., None], local, parking)
    changed=(active!=env._active_object_mask[ids]).any(-1)
    if bool(changed.any()):
        # Preserve the surface observer's target-only fallback, without physics
        # steps or writes to environments outside this partial reset.
        world=shelf_state[:,None,:3]+quat_apply(q,local)
        states[:,:,:3]=world
        rotation=relative.new_zeros(relative.shape)
        rotation[:,:,0]=(yaw/2).cos();rotation[:,:,3]=(yaw/2).sin()
        states[:,:,3:7]=quat_mul(q,rotation)
        objects.write_object_link_state_to_sim(states[changed],env_ids=ids[changed])
        env._active_object_mask[ids]=active
        apply_fixed_pool_gravity_mask(env,ids[changed])
        env._revision_spawn_positions[ids]=world
    env._surface_target_w[ids]=targets
    env._surface_obstacles_w[ids]=obstacles
    env._surface_valid[ids]=valid
    actual=active[:,1:].sum(-1)
    env._final_actual_population[ids]=actual
    env._reset_diagnostics={
        'Reset/requested_obstacles':requested.float().mean(),
        'Reset/actual_obstacles':actual.float().mean(),
        'Reset/placement_skipped':0.,
        'Reset/empty_fraction':(actual==0).float().mean(),
        'Reset/ik_attempts':1.,
        'Reset/layout_resamples':0.,
        'Reset/ik_recovery_fraction':0.,
        'Reset/finger_placement_ms':placement_ms,
        'Reset/observation_ms':sum(t['total_ms'] for t in timings)/len(ids),
        **{f'Reset/observation_{key}':sum(t.get(key,0.) for t in timings)/len(ids)
           for key in ('upload_ms','readback_ms','ray_ms','components_ms')},
        'Reset/observation_fallbacks':sum(t['fallback'] for t in timings)/len(ids),
        'Reset/target_missing_fallbacks':sum(t['target_missing'] for t in timings)/len(ids),
        'Reset/slot_overflow_fallbacks':sum(t['slot_overflow'] for t in timings)/len(ids),
        'Reset/batch_ms':1000*(perf_counter()-start),
    }
