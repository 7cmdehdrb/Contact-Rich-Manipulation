"""Reset-only physics randomization and reusable, per-environment initial states."""
import math
import torch
from .wrench_refactor import initialize_physics
from ..common.realistic_physics import MASS_LIMITS


def randomize_reset_physics(env, env_ids, mass_scale=1.5):
    """Keep the parent's uniform distributions; update only reset view rows.

    All 17 bodies draw independently. Inertia always scales from immutable
    geometry-specific startup values, never from the previous episode.
    Shelf, object and hand friction use the parent's finite material buckets.
    """
    initialize_physics(env, None, mass_scale=mass_scale)
    ids = (torch.arange(env.num_envs) if env_ids is None else
           torch.as_tensor(env_ids, device='cpu', dtype=torch.long))
    if not ids.numel():
        return
    if not hasattr(env, '_reset_physics_rng'):
        env._reset_physics_rng = torch.Generator().manual_seed(int(env.cfg.seed or 0) + 173)
    rng = env._reset_physics_rng
    state = env._clutter_physics
    objects = env.scene['object_collection']
    rows = (torch.arange(objects.num_objects)[:, None] * env.num_envs + ids[None]).flatten()
    mass = (MASS_LIMITS[0] + torch.rand((len(rows), 1), generator=rng)
            * (MASS_LIMITS[1] - MASS_LIMITS[0])) * mass_scale
    state.mass_buffer[rows] = mass
    state.inertia_buffer[rows] = state.default_inertia[rows] * (mass / state.default_mass[rows])
    objects.root_physx_view.set_masses(state.mass_buffer, rows)
    objects.root_physx_view.set_inertias(state.inertia_buffer, rows)
    state.mass[ids] = mass[:len(ids), 0]
    for role, material in enumerate(state.materials):
        selected = rows if role == 1 else ids
        bucket = torch.randint(state.friction_buckets.shape[1], (len(selected),), generator=rng)
        pairs = state.friction_buckets[role, bucket]
        material.apply(selected, pairs[:, 0], pairs[:, 1])
        state.static_friction[ids, role] = pairs[:len(ids), 0]
        state.dynamic_friction[ids, role] = pairs[:len(ids), 1]


def _configuration_key(value):
    if isinstance(value, dict):
        return tuple((k, _configuration_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_configuration_key(v) for v in value)
    return value


def grouped_reset_scene(env, env_ids, ik_params, spawn_radius=.02, edge_clearance=.02,
                        excluded_x_half_angle=math.pi / 6, min_goal_distance=.06,
                        max_goal_distance=.12, observation_noise=.01,
                        visibility_threshold=.35, visibility_flip_probability=.01,
                        observation_swap_probability=.1, sweep_direction='both',
                        goal_angle_range=None, fixed_goal_offset=None):
    """Share compatible initial states, refreshing them without changing physics steps.

    Only env_ids are written. Each participating size group selects one state;
    misses are generated together using the parent's layout and IK algorithms.
    Geometry-dependent state is reused; visual noise and physics are not cached.
    """
    from .wrench_refactor import reset_scene, reset_robot_from_goal_ik, apply_fixed_pool_gravity_mask
    from .final import apply_final_visibility
    from ..common.grouped_reset_pool import GroupedResetPool
    from isaaclab.utils.math import subtract_frame_transforms

    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None else
           torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if not len(ids):
        return
    scene_params = dict(spawn_radius=spawn_radius, edge_clearance=edge_clearance,
                        excluded_x_half_angle=excluded_x_half_angle,
                        min_goal_distance=min_goal_distance, max_goal_distance=max_goal_distance,
                        observation_noise=observation_noise, visibility_threshold=visibility_threshold,
                        visibility_flip_probability=visibility_flip_probability,
                        observation_swap_probability=observation_swap_probability,
                        sweep_direction=sweep_direction, goal_angle_range=goal_angle_range,
                        fixed_goal_offset=fixed_goal_offset)
    groups = min(env.cfg.reset_size_groups, env.num_envs)
    if not hasattr(env, '_grouped_reset_pool'):
        env._grouped_reset_pool = GroupedResetPool(
            groups, env.cfg.reset_pool_capacity, env.cfg.reset_pool_refresh_interval,
            getattr(env, '_reset_pool_seed', int(env.cfg.seed or 0)) + 2717)
    pool = env._grouped_reset_pool
    key = _configuration_key((scene_params, ik_params, env.cfg.red_margin, env.cfg.rear_offset))
    # One host transfer for bookkeeping; no per-object GPU readback.
    host_ids = ids.cpu().tolist()
    participating = {}
    for env_id in host_ids:
        participating.setdefault(env_id % groups, env_id)
    selected, missing = {}, []
    for group, representative in participating.items():
        slot, fresh = pool.select(group, key)
        selected[group] = slot
        if fresh:
            missing.append((group, slot, representative))
    if missing:
        representatives = torch.tensor([r for _, _, r in missing], device=env.device)
        reset_scene(env, representatives, **scene_params)
        reset_robot_from_goal_ik(env, representatives, **ik_params)
        robot = env.scene['robot']
        state = env.scene['object_collection'].data.object_link_state_w
        for group, slot, row in missing:
            origin = env.scene.env_origins[row]
            pose = state[row].clone()
            pose[:, :3] -= origin
            entry = dict(states=pose, active=env._active_object_mask[row].clone(),
                         delta=env.sweep_dir[row].clone(), width=env.target_width[row].clone(),
                         actual=env._final_actual_population[row].clone(),
                         joints=robot.data.joint_pos[row, env._goal_reset_arm_ids].clone(),
                         desired_pos=env._goal_reset_desired_pos_w[row].clone() - origin,
                         desired_quat=env._goal_reset_desired_quat_w[row].clone(),
                         errors=env._goal_reset_errors[row].clone())
            if env.cfg.debug_initial_ik_bounds:
                entry.update(marker_pos=env._ik_bounds_positions[row].clone() - origin,
                             marker_rot=env._ik_bounds_rotations[row].clone(),
                             marker_scale=env._ik_bounds_scales[row].clone())
            pool.entries[group][slot] = entry
    entries = [pool.entries[group][selected[group]] for group in participating]
    group_index = {g:i for i,g in enumerate(participating)}
    mapping = torch.tensor([group_index[i % groups] for i in host_ids], device=env.device)
    def gather(name):
        return torch.stack([entry[name] for entry in entries])[mapping]

    origin = env.scene.env_origins[ids]
    states = gather('states')
    states[..., :3] += origin[:, None]
    states[..., 7:] = 0.
    env.scene['object_collection'].write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = gather('active')
    apply_fixed_pool_gravity_mask(env, ids)
    env.target_id[ids, 0] = 0
    env.target_width[ids] = gather('width')
    env.sweep_dir[ids] = gather('delta')
    env._final_actual_population[ids] = gather('actual')
    env._revision_layout_complete[ids] = True
    env._revision_spawn_positions[ids] = states[..., :3]
    env.sensor_data_fresh[ids] = False

    robot = env.scene['robot']
    q = gather('joints')
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=env._goal_reset_arm_ids, env_ids=ids)
    robot.set_joint_position_target(q, joint_ids=env._goal_reset_arm_ids, env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(q), joint_ids=env._goal_reset_arm_ids, env_ids=ids)
    _ = robot.data.body_link_pos_w
    env.scene['ee_frame'].reset(ids)
    env._goal_reset_desired_pos_w[ids] = gather('desired_pos') + origin
    env._goal_reset_desired_quat_w[ids] = gather('desired_quat')
    env._goal_reset_errors[ids] = gather('errors')

    # Independently redraw every environment's initial visual estimate.
    positions, _ = subtract_frame_transforms(
        robot.data.root_pos_w[ids, None].expand(-1, 16, -1),
        robot.data.root_quat_w[ids, None].expand(-1, 16, -1), states[:, 1:, :3])
    positions += (2 * torch.rand(positions.shape, device=env.device, generator=env._revision_layout_rng)-1) * scene_params.get('observation_noise', .01)
    active = env._active_object_mask[ids]
    env._initial_obstacle_observation[ids] = torch.cat(
        (active[:, 1:, None].float(), torch.where(active[:, 1:, None], positions, 0.)), -1).flatten(1)
    apply_final_visibility(env, ids, scene_params.get('visibility_threshold', .35),
                           scene_params.get('visibility_flip_probability', .01),
                           scene_params.get('observation_swap_probability', .1))
    if env.cfg.debug_initial_ik_bounds:
        env._ik_bounds_positions[ids] = gather('marker_pos') + origin[:, None]
        env._ik_bounds_rotations[ids] = gather('marker_rot')
        env._ik_bounds_scales[ids] = gather('marker_scale')
        env._ik_bounds_active[ids] = True
        mask = env._ik_bounds_active
        indices = torch.arange(24, device=env.device).div(12, rounding_mode='floor')
        env._initial_ik_bounds_marker.visualize(
            translations=env._ik_bounds_positions[mask].reshape(-1, 3),
            orientations=env._ik_bounds_rotations[mask].reshape(-1, 4),
            scales=env._ik_bounds_scales[mask].reshape(-1, 3),
            marker_indices=indices.repeat(int(mask.sum())))
