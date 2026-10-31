"""Final-only size randomization and failure-tolerant reset placement."""

import math
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, subtract_frame_transforms
from ..common.final_geometry import sample_target_goal, sample_population
from ..common.contact_sweep_clutter_fusion_population import PARKING_HEIGHT, active_population_mask
from .contact_sweep_clutter_fusion import apply_gravity_mask, classify_visibility_torch, pack_obstacle_observations


def apply_final_visibility(env, ids, visibility_threshold, visibility_flip_probability, observation_swap_probability):
    """Use each scaled object's conservative footprint for initial occlusion."""
    objects = env.scene['object_collection']
    shelf = env.scene['shelf'].data
    position = quat_apply_inverse(shelf.root_quat_w[ids, None].expand(-1, objects.num_objects, -1),
                                 objects.data.object_pos_w[ids] - shelf.root_pos_w[ids, None])
    centers = torch.stack((position[..., 1], -position[..., 0]), dim=-1)
    bounds = env._final_object_bounds[ids.cpu().numpy()]
    half = np.max(np.abs(bounds[..., :2]), axis=2)
    radii = torch.as_tensor(np.linalg.norm(half, axis=-1), device=env.device, dtype=centers.dtype)
    active = env._active_object_mask[ids]
    visible = classify_visibility_torch(centers, radii=radii, active=active,
                                       threshold=visibility_threshold, check_inputs=False).visible[:, 1:]
    flips = torch.rand(visible.shape, device=env.device) < visibility_flip_probability
    visible = torch.logical_xor(visible, flips) & active[:, 1:]
    slots = env._initial_obstacle_observation[ids].reshape(len(ids), -1, 4)
    capacity = objects.num_objects - 1
    slots[:, :capacity] = torch.where(visible[..., None], slots[:, :capacity], 0.)
    slots[:, capacity:] = 0.
    env._initial_obstacle_observation[ids] = pack_obstacle_observations(slots, observation_swap_probability).flatten(1)


def randomize_object_sizes(env, env_ids, scale_range=(.9, 1.1)):
    """USD changes only before PhysX starts; scale visual and collider together.

    Shape and size are sampled per vectorized environment at construction.
    Reset randomizes placement/physics while retaining the cooked geometry.
    """
    from pxr import Gf, UsdGeom, Usd, UsdPhysics
    import isaaclab.sim as sim

    if env.sim.is_playing():
        raise RuntimeError('Object size randomization must run in prestartup mode')
    if not 0 < scale_range[0] <= scale_range[1]:
        raise ValueError('Expected positive ordered scale range')
    ids = range(env.num_envs) if env_ids is None else torch.as_tensor(env_ids).cpu().tolist()
    names = list(env.cfg.scene.object_collection.rigid_objects)
    rng = np.random.default_rng(env.cfg.seed)
    bounds = np.zeros((env.num_envs, len(names), 2, 3))
    scales = np.ones((env.num_envs, len(names)))
    stage = sim.get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for row in ids:
        for index, name in enumerate(names):
            root = stage.GetPrimAtPath(f'/World/envs/env_{row}/{name}')
            # Local bound before adding root scale includes bottom-origin mesh
            # translations for both standard cylinders and scanned props.
            box = cache.ComputeUntransformedBound(root).ComputeAlignedRange()
            scale = float(rng.uniform(*scale_range))
            xform = UsdGeom.Xformable(root)
            op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
            if op is None:
                op = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
            op.Set(Gf.Vec3d(scale, scale, scale))
            bounds[row, index] = np.array([tuple(box.GetMin()), tuple(box.GetMax())]) * scale
            scales[row, index] = scale
            # Explicit inertia is not geometrically scaled by PhysX. COM is
            # in prim-local coordinates and PhysX applies the root scale, so
            # leave COM unchanged to avoid scaling it twice.
            mass = UsdPhysics.MassAPI(root)
            if mass:
                inertia = mass.GetDiagonalInertiaAttr().Get()
                if inertia is not None and any(v > 0 for v in inertia):
                    mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*(float(v) * scale ** 2 for v in inertia)))
    env._final_object_bounds = bounds
    env._final_object_scales = scales
    if env.cfg.target_bounding_box:
        from ...assets.final_visualization import create_target_bounding_boxes
        create_target_bounding_boxes(env)


def randomize_final_scene(
    env, env_ids, spawn_radius=.02, goal_distance=.15, object_radius=.04,
    edge_clearance=.02, shelf_half_extents=(.20, .50), shelf_surface_height=1.05,
    excluded_x_half_angle=math.radians(30), observation_noise=.01,
    visibility_threshold=.35, visibility_flip_probability=.01, observation_swap_probability=.1,
    gray_obstacles=1, white_obstacles=1, gray_red_padding_fraction=0., wide_target=False,
):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
    if not len(ids):
        return
    if not 0 <= gray_red_padding_fraction <= .15:
        raise ValueError('Gray padding fraction must be in [0, .15]')
    if not hasattr(env, '_final_layout_rng'):
        env._final_layout_rng = np.random.default_rng(env._population_layout_seed)
    rng = env._final_layout_rng
    objects = env.scene['object_collection']
    max_gray, max_white = env.cfg.max_gray_obstacles, env.cfg.max_white_obstacles
    gray, white = min(gray_obstacles, max_gray), min(white_obstacles, max_white)
    count, capacity = len(ids), max_gray + max_white
    local = np.zeros((count, capacity + 1, 3))
    local[..., 2] = PARKING_HEIGHT
    local[..., 1] = np.arange(capacity + 1) * .2
    yaws, displacements, masks, actual_counts = [], [], [], []
    for row, env_id in enumerate(ids.cpu().tolist()):
        bounds = env._final_object_bounds[env_id]
        half = np.max(np.abs(bounds[:, :, :2]), axis=1)
        target, goal, config = sample_target_goal(rng, half[0], wide_target, spawn_radius,
            goal_distance, edge_clearance, excluded_x_half_angle, shelf_half_extents)
        centers, yaw, actual_gray, actual_white = sample_population(
            rng, target, goal, half, max_gray, gray, white, config, gray_red_padding_fraction)
        active_indices = [0, *range(1, 1 + actual_gray), *range(1 + max_gray, 1 + max_gray + actual_white)]
        local[row, active_indices, :2] = np.concatenate((target[None], centers))
        # Actual lower bound accounts for every size/shape and prevents penetration.
        local[row, active_indices, 2] = shelf_surface_height - bounds[active_indices, 0, 2]
        masks.append(active_population_mask(1, max_gray, max_white, actual_gray, actual_white,
                                            env.device, env.cfg.obstacle_observation_slots)[0])
        yaws.append(yaw)
        displacements.append(goal - target)
        actual_counts.append((actual_gray, actual_white))
    local = torch.as_tensor(local, dtype=torch.float32, device=env.device)
    yaw = torch.as_tensor(np.asarray(yaws), dtype=torch.float32, device=env.device)
    active = torch.stack(masks)
    shelf = env.scene['shelf'].data
    rotation = shelf.root_quat_w[ids, None].expand(-1, capacity + 1, -1)
    world = shelf.root_pos_w[ids, None] + quat_apply(rotation, local)
    yaw_rotation = torch.zeros((count, capacity + 1, 4), device=env.device)
    yaw_rotation[..., 0], yaw_rotation[..., 3] = (yaw / 2).cos(), (yaw / 2).sin()
    states = torch.cat((world, quat_mul(rotation, yaw_rotation), torch.zeros((count, capacity + 1, 6), device=env.device)), -1)
    objects.write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = active
    apply_gravity_mask(env, ids)
    env.target_id[ids, 0] = 0
    target_half = np.max(np.abs(env._final_object_bounds[ids.cpu().numpy(), 0, :, :2]), axis=1)
    env.target_width[ids, 0] = torch.as_tensor(np.linalg.norm(target_half, axis=-1), device=env.device, dtype=torch.float32)
    displacement = torch.zeros((count, 3), device=env.device)
    displacement[:, :2] = torch.as_tensor(np.asarray(displacements), device=env.device, dtype=torch.float32)
    env.sweep_dir[ids] = quat_apply(shelf.root_quat_w[ids], displacement)
    robot = env.scene['robot'].data
    positions, _ = subtract_frame_transforms(robot.root_pos_w[ids, None].expand(-1, capacity, -1),
        robot.root_quat_w[ids, None].expand(-1, capacity, -1), world[:, 1:])
    positions += torch.empty_like(positions).uniform_(-observation_noise, observation_noise)
    slots = torch.zeros((count, env.cfg.obstacle_observation_slots, 4), device=env.device)
    slots[:, :capacity, 0] = active[:, 1:]
    slots[:, :capacity, 1:] = torch.where(active[:, 1:, None], positions, 0.)
    env._initial_obstacle_observation[ids] = slots.flatten(1)
    apply_final_visibility(env, ids, visibility_threshold, visibility_flip_probability, observation_swap_probability)
    if not hasattr(env, '_final_actual_population'):
        env._final_actual_population = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)
    env._final_actual_population[ids] = torch.as_tensor(actual_counts, device=env.device)


def randomize_final_target_scene(
    env, env_ids, spawn_radius=.02, goal_distance=.15, object_radius=.04,
    edge_clearance=.02, shelf_half_extents=(.20, .50), shelf_surface_height=1.05,
    excluded_x_half_angle=math.radians(30), observation_noise=.01,
    visibility_threshold=.35, visibility_flip_probability=.01, observation_swap_probability=.1,
    gray_obstacles=1, white_obstacles=1, gray_red_padding_fraction=0., wide_target=False,
    gray_padding=.06, finger_gap=.02, placement_candidates=32, placement_restarts=4, placement_passes=2,
):
    """v0_2_1: select cooked target geometry, then perform batched OBB placement."""
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    pool = env.scene['object_collection']
    choices = env._target_choice.sample(len(ids))
    pool.switch(ids, choices, env.scene.env_origins)
    cpu_ids = ids.cpu().numpy()
    physical_ids = pool.mapping[ids, 0].cpu().numpy()
    env._final_object_bounds[cpu_ids, 0] = env._target_pool_bounds[cpu_ids, physical_ids]
    env._final_object_scales[cpu_ids, 0] = env._target_pool_scales[cpu_ids, physical_ids]
    env._final_object_bounds_t[ids, 0] = env._target_pool_bounds_t[ids, pool.mapping[ids, 0]]
    # Parent physics events must scale the newly selected shape's original inertia.
    physics = env._clutter_physics
    rows = torch.as_tensor(physical_ids * env.num_envs + cpu_ids)
    physics.default_mass[cpu_ids] = pool.default_pool_mass[rows]
    physics.default_inertia[cpu_ids] = pool.default_pool_inertia[rows]
    env.sensor_data_fresh[ids] = False
    from ..common.final_obb_geometry import sample_shelf_layout
    local, yaw, active, delta, actual = sample_shelf_layout(
        env._final_object_bounds_t[ids], env._obb_layout_rng,
        max_gray_slots=env.cfg.max_gray_obstacles, max_white_slots=env.cfg.max_white_obstacles,
        gray_obstacles=gray_obstacles, white_obstacles=white_obstacles,
        gray_padding=gray_padding, finger_gap=finger_gap, gray_red_padding_fraction=gray_red_padding_fraction,
        placement_candidates=placement_candidates, placement_restarts=placement_restarts, placement_passes=placement_passes,
        spawn_radius=spawn_radius, goal_distance=goal_distance, edge_clearance=edge_clearance,
        shelf_half_extents=shelf_half_extents, shelf_surface_height=shelf_surface_height,
        excluded_x_half_angle=excluded_x_half_angle, wide_target=wide_target,
    )
    # Preserve the parent's state, gravity, command and observation contracts.
    count, capacity = len(ids), pool.num_objects-1
    shelf = env.scene['shelf'].data
    rotation = shelf.root_quat_w[ids, None].expand(-1, capacity+1, -1)
    world = shelf.root_pos_w[ids, None] + quat_apply(rotation, local)
    yaw_rotation = torch.zeros((count, capacity+1, 4), device=env.device)
    yaw_rotation[..., 0], yaw_rotation[..., 3] = (yaw/2).cos(), (yaw/2).sin()
    states = torch.cat((world, quat_mul(rotation, yaw_rotation),
                        torch.zeros((count, capacity+1, 6), device=env.device)), -1)
    pool.write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = active
    apply_gravity_mask(env, ids)
    env.target_id[ids, 0] = 0
    target_half = env._final_object_bounds_t[ids, 0, :, :2].abs().amax(1)
    env.target_width[ids, 0] = torch.linalg.vector_norm(target_half, dim=-1)
    displacement = torch.zeros((count, 3), device=env.device)
    displacement[:, :2] = delta
    env.sweep_dir[ids] = quat_apply(shelf.root_quat_w[ids], displacement)
    robot = env.scene['robot'].data
    positions, _ = subtract_frame_transforms(robot.root_pos_w[ids, None].expand(-1, capacity, -1),
        robot.root_quat_w[ids, None].expand(-1, capacity, -1), world[:, 1:])
    positions += torch.empty_like(positions).uniform_(-observation_noise, observation_noise)
    slots = torch.zeros((count, env.cfg.obstacle_observation_slots, 4), device=env.device)
    slots[:, :capacity, 0] = active[:, 1:]
    slots[:, :capacity, 1:] = torch.where(active[:, 1:, None], positions, 0.)
    env._initial_obstacle_observation[ids] = slots.flatten(1)
    apply_final_visibility(env, ids, visibility_threshold, visibility_flip_probability, observation_swap_probability)
    if not hasattr(env, '_final_actual_population'):
        env._final_actual_population = torch.zeros((env.num_envs, 2), dtype=torch.long, device=env.device)
    env._final_actual_population[ids] = actual
