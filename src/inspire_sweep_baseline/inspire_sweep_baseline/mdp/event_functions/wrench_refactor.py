"""Fixed geometry/material lifecycle; reset only poses, activity and snapshots."""
import math
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms
from ..common.wrench_refactor_layout import fixed_pool_layout
from .final import apply_final_visibility
from .contact_sweep_clutter_physics import initialize_clutter_materials


def apply_fixed_pool_gravity_mask(env, ids):
    """Update gravity flags and wake every changed row, including parked bodies."""
    objects = env.scene['object_collection']
    rows = (torch.arange(objects.num_objects)[:, None] * env.num_envs + ids.cpu()[None]).flatten()
    flags = (~env._active_object_mask[ids]).T.reshape(-1, 1).to(device='cpu', dtype=torch.uint8)
    env._wrench_gravity_flags[rows] = flags
    objects.root_physx_view.set_disable_gravities(env._wrench_gravity_flags, rows)
    objects.root_physx_view.wake_up(rows)


def initialize_geometry(env, env_ids):
    """Read spawn metadata and compute scale-correct bottom alignment once."""
    from ...assets import revision_spawners as assets
    if env.sim.is_playing():
        raise RuntimeError('Fixed geometry must be initialized before physics starts')
    names = tuple(env.cfg.scene.object_collection.rigid_objects)
    if names != ('mass_cylinder', *(f'obstacle_{i}' for i in range(16))):
        raise ValueError('Invalid Refactor fixed pool')
    metadata = assets._prototype_cache['metadata']
    values = [[metadata[f'/World/envs/env_{i}/{name}'] for name in names] for i in range(env.num_envs)]
    env._final_object_bounds = np.asarray([[v[2] for v in row] for row in values])
    env._final_object_scales = np.asarray([[v[1] for v in row] for row in values])
    env._revision_object_kinds = np.asarray([[v[0] for v in row] for row in values])
    env._revision_bounds_t = torch.as_tensor(env._final_object_bounds, device=env.device, dtype=torch.float32)
    env._refactor_spawn_z = 1.05 - env._revision_bounds_t[..., 0, 2]
    env._wrench_gravity_flags = torch.ones((env.num_envs * 17, 1), dtype=torch.uint8)


def initialize_physics(env, env_ids, mass_scale=1.5):
    """Uniform startup draws, fixed across resets; finite shared material buckets.

    Keep prior supported mass endpoints and mean +/- two standard deviations
    for friction, replacing Gaussian sampling with uniform sampling. Every
    object gets its own mass/material; floor and hand get per-environment draws.
    """
    if getattr(env, '_refactor_physics_initialized', False):
        return
    from ..common.realistic_physics import (MASS_LIMITS, STATIC_FRICTION_MEANS,
        STATIC_FRICTION_STDS, DYNAMIC_FRICTION_MEANS, DYNAMIC_FRICTION_STDS)
    initialize_clutter_materials(env)
    state = env._clutter_physics
    objects = env.scene['object_collection']
    count = env.num_envs
    rng = torch.Generator().manual_seed(int(env.cfg.seed or 0) + 173)
    masses = (MASS_LIMITS[0] + torch.rand((count * 17, 1), generator=rng)
              * (MASS_LIMITS[1] - MASS_LIMITS[0])) * mass_scale
    rows = torch.arange(count * 17)
    state.mass_buffer[:] = masses
    state.inertia_buffer[:] = state.default_inertia * (masses / state.default_mass)
    objects.root_physx_view.set_masses(state.mass_buffer, rows)
    objects.root_physx_view.set_inertias(state.inertia_buffer, rows)
    pools = []
    for mean_s, std_s, mean_d, std_d in zip(STATIC_FRICTION_MEANS, STATIC_FRICTION_STDS,
                                           DYNAMIC_FRICTION_MEANS, DYNAMIC_FRICTION_STDS):
        u = (torch.arange(4096) + .5) / 4096
        static = max(0., mean_s - 2 * std_s) + u * (mean_s + 2 * std_s - max(0., mean_s - 2 * std_s))
        dynamic = max(0., mean_d - 2 * std_d) + u[torch.randperm(4096, generator=rng)] * (mean_d + 2 * std_d - max(0., mean_d - 2 * std_d))
        pools.append(torch.stack((static, dynamic), -1))
    state.friction_buckets = torch.stack(pools)
    for role, material in enumerate(state.materials):
        size = count * 17 if role == 1 else count
        pairs = pools[role][torch.randint(4096, (size,), generator=rng)]
        material.apply(torch.arange(size), pairs[:, 0], pairs[:, 1])
        state.static_friction[:, role] = pairs[:count, 0]
        state.dynamic_friction[:, role] = pairs[:count, 1]
    state.mass[:] = masses[:count, 0]
    env._refactor_physics_initialized = True


def reset_scene(env, env_ids, spawn_radius=.02, edge_clearance=.02,
                excluded_x_half_angle=math.pi / 6, min_goal_distance=.06, max_goal_distance=.12,
                observation_noise=.01,
                visibility_threshold=.35, visibility_flip_probability=.01,
                observation_swap_probability=.1, sweep_direction='both', goal_angle_range=None,
                fixed_goal_offset=None):
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if not len(ids):
        return
    counts = torch.randint(0, 4, (len(ids), 2), device=env.device, generator=env._revision_layout_rng)
    local, yaw, active, delta, actual = fixed_pool_layout(
        env._revision_bounds_t[ids], env._refactor_spawn_z[ids], env._revision_layout_rng, counts,
        red_margin=env.cfg.red_margin, rear_offset=env.cfg.rear_offset,
        spawn_radius=spawn_radius, edge_clearance=edge_clearance,
        excluded_x_half_angle=excluded_x_half_angle, min_goal_distance=min_goal_distance,
        max_goal_distance=max_goal_distance, sweep_direction=sweep_direction,
        goal_angle_range=goal_angle_range, fixed_goal_offset=fixed_goal_offset)
    shelf = env.scene['shelf'].data
    rotation = shelf.root_quat_w[ids, None].expand(-1, 17, -1)
    world = shelf.root_pos_w[ids, None] + quat_apply(rotation, local)
    yaw_rotation = torch.zeros((len(ids), 17, 4), device=env.device)
    yaw_rotation[..., 0], yaw_rotation[..., 3] = (yaw / 2).cos(), (yaw / 2).sin()
    states = torch.cat((world, quat_mul(rotation, yaw_rotation), world.new_zeros(len(ids), 17, 6)), -1)
    env.scene['object_collection'].write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = active
    apply_fixed_pool_gravity_mask(env, ids)
    env.target_id[ids, 0] = 0
    half = env._revision_bounds_t[ids, 0, :, :2].abs().amax(1)
    env.target_width[ids, 0] = half.norm(dim=-1)
    env.sweep_dir[ids] = quat_apply(shelf.root_quat_w[ids], torch.cat((delta, delta.new_zeros(len(ids), 1)), -1))
    robot = env.scene['robot'].data
    positions, _ = subtract_frame_transforms(robot.root_pos_w[ids, None].expand(-1, 16, -1),
        robot.root_quat_w[ids, None].expand(-1, 16, -1), world[:, 1:])
    positions += (2 * torch.rand(positions.shape, device=env.device, generator=env._revision_layout_rng) - 1) * observation_noise
    slots = torch.cat((active[:, 1:, None].float(), torch.where(active[:, 1:, None], positions, 0.)), -1)
    env._initial_obstacle_observation[ids] = slots.flatten(1)
    apply_final_visibility(env, ids, visibility_threshold, visibility_flip_probability, observation_swap_probability)
    env._final_actual_population[ids] = actual
    env._revision_layout_complete[ids] = True
    env._revision_spawn_positions[ids] = world
    env.sensor_data_fresh[ids] = False


def reset_robot_from_goal_ik(env, env_ids, front_clearance=(0.10, 0.13),
                            lateral_offset=(0.10, 0.14), height_jitter=0.01,
                            height_offset=0.0, max_condition=None, joint_margin=0.0,
                            max_attempts=1):
    """Run after object_spawn and before observation/action/reward manager resets."""
    from isaaclab.sim.utils import get_current_stage
    from isaaclab.utils.math import quat_apply_inverse, quat_mul
    from ...assets.robot import ARM_JOINT_NAMES
    from ..common.wrench_refactor_ik import ResetArmIK, sample_initial_eef

    if (not math.isfinite(height_offset) or max_attempts < 1
            or (max_condition is not None and (not math.isfinite(max_condition) or max_condition < 1))
            or not math.isfinite(joint_margin) or joint_margin < 0):
        raise ValueError('Invalid initial IK height/quality/retry settings')

    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if not len(ids):
        return
    robot, shelf = env.scene['robot'], env.scene['shelf'].data
    if not hasattr(env, '_goal_reset_ik'):
        offset = env.cfg.scene.ee_frame.target_frames[0].offset
        path = env.scene.env_prim_paths[0] + '/Robot'
        env._goal_reset_ik = ResetArmIK.from_stage(get_current_stage(), path, ARM_JOINT_NAMES,
                                                 offset.pos, offset.rot, env.device)
        env._goal_reset_arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        env._goal_reset_base_id = robot.find_bodies('base_link')[0][0]
        env._goal_reset_rng = torch.Generator(device=env.device).manual_seed(
            getattr(env, '_goal_reset_seed', int(env.cfg.seed or 0) + 1707))
        env._goal_reset_desired_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
        env._goal_reset_desired_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
        env._goal_reset_errors = torch.zeros((env.num_envs, 2), device=env.device)
    chain, arm_ids = env._goal_reset_ik, env._goal_reset_arm_ids
    base_pos = robot.data.body_link_pos_w[ids, env._goal_reset_base_id].clone()
    base_quat = robot.data.body_link_quat_w[ids, env._goal_reset_base_id].clone()
    initial = robot.data.default_joint_pos[ids][:, arm_ids]
    default_pos, default_quat = chain.forward(initial)
    default_world = base_pos + quat_apply(base_quat, default_pos)
    default_local = quat_apply_inverse(shelf.root_quat_w[ids], default_world - shelf.root_pos_w[ids])
    default_local[:, 2] += height_offset
    target_local = quat_apply_inverse(shelf.root_quat_w[ids],
        env._revision_spawn_positions[ids, 0] - shelf.root_pos_w[ids])
    sweep_local = quat_apply_inverse(shelf.root_quat_w[ids], env.sweep_dir[ids])
    desired_local = sample_initial_eef(target_local, sweep_local, default_local, env._goal_reset_rng,
                                       front_clearance, lateral_offset, height_jitter)
    desired_world = shelf.root_pos_w[ids] + quat_apply(shelf.root_quat_w[ids], desired_local)
    desired_base, _ = subtract_frame_transforms(base_pos, base_quat, desired_world)
    limits = robot.data.joint_pos_limits[ids][:, arm_ids]
    angles, valid, position_error, rotation_error = chain.solve(desired_base, default_quat, initial, limits)
    if max_condition is not None:
        # Normalize translation by a 0.3 m characteristic arm length before SVD.
        # Otherwise a mixed metre/radian condition number depends on units.
        from ..common.wrench_refactor_ik import initial_pose_quality
        for attempt in range(max_attempts):
            quality, condition, margin = initial_pose_quality(chain, angles, limits, max_condition, joint_margin)
            valid &= quality
            if bool(valid.all()) or attempt == max_attempts - 1:
                break
            retry = ~valid
            desired_local[retry] = sample_initial_eef(
                target_local[retry], sweep_local[retry], default_local[retry], env._goal_reset_rng,
                front_clearance, lateral_offset, height_jitter)
            desired_world[retry] = shelf.root_pos_w[ids][retry] + quat_apply(
                shelf.root_quat_w[ids][retry], desired_local[retry])
            retry_base, _ = subtract_frame_transforms(base_pos[retry], base_quat[retry], desired_world[retry])
            angles[retry], valid[retry], position_error[retry], rotation_error[retry] = chain.solve(
                retry_base, default_quat[retry], initial[retry], limits[retry])
        if not hasattr(env, '_goal_reset_condition'):
            env._goal_reset_condition = torch.zeros(env.num_envs, device=env.device)
            env._goal_reset_joint_margin = torch.zeros(env.num_envs, device=env.device)
        env._goal_reset_condition[ids] = condition
        env._goal_reset_joint_margin[ids] = margin
    if not bool(valid.all()):
        raise RuntimeError(f'Goal-conditioned reset IK pose/quality check failed for envs {ids[~valid].tolist()}; '
                           f'max position error={position_error.max().item():.6f}m, '
                           f'rotation error={rotation_error.max().item():.6f}rad')
    # No fallback to a default pose on the wrong side. No intermediate IK
    # iterates are written to PhysX, and unrelated environments never advance.
    robot.write_joint_state_to_sim(angles, torch.zeros_like(angles), joint_ids=arm_ids, env_ids=ids)
    robot.set_joint_position_target(angles, joint_ids=arm_ids, env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(angles), joint_ids=arm_ids, env_ids=ids)
    # Refresh PhysX link transforms before snapshot observations/rewards read
    # the frame sensor in their reset callbacks (without advancing physics).
    _ = robot.data.body_link_pos_w
    env.scene['ee_frame'].reset(ids)
    env._goal_reset_desired_pos_w[ids] = desired_world
    env._goal_reset_desired_quat_w[ids] = quat_mul(base_quat, default_quat)
    env._goal_reset_errors[ids] = torch.stack((position_error, rotation_error), -1)
    if getattr(env.cfg, 'debug_initial_ik_bounds', False):
        env._update_initial_ik_bounds(ids, target_local, default_local,
                                     shelf.root_pos_w[ids], shelf.root_quat_w[ids],
                                     front_clearance, lateral_offset, height_jitter)
