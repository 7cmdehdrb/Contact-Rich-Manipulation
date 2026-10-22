"""Reset fixed target/goal distributions and the current obstacle population."""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, subtract_frame_transforms

from ..contact_sweep_clutter_fusion.events import classify_visibility_torch
from ..mdp.goal_events import _goal_ray_distance, _sample_planar_unit_vectors, _sample_uniform_disk
from .layout import generator_cm_to_shelf_xy, population_layout_generator, shelf_xy_to_generator_cm
from .population import PARKING_HEIGHT, active_population_mask, apply_gravity_mask


def apply_population_visibility(
    env, env_ids, object_radius=0.04, visibility_threshold=0.35, visibility_flip_probability=0.01,
):
    """Inactive capacity neither occludes nor appears in the initial snapshot."""
    if not math.isfinite(object_radius) or object_radius <= 0:
        raise ValueError("object_radius must be finite and positive")
    if not math.isfinite(visibility_threshold) or not 0 < visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be in (0, 1]")
    if not math.isfinite(visibility_flip_probability) or not 0 <= visibility_flip_probability <= 1:
        raise ValueError("visibility_flip_probability must be in [0, 1]")
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    if not ids.numel():
        return
    objects = env.scene["object_collection"]
    shelf = env.scene["shelf"].data
    shelf_positions = quat_apply_inverse(
        shelf.root_quat_w[ids, None].expand(-1, objects.num_objects, -1),
        objects.data.object_pos_w[ids] - shelf.root_pos_w[ids, None],
    )
    centers = torch.stack((shelf_positions[..., 1], -shelf_positions[..., 0]), dim=-1)
    active = env._active_object_mask[ids]
    visible = classify_visibility_torch(
        centers, radii=object_radius, active=active, threshold=visibility_threshold, check_inputs=False,
    ).visible[:, 1:]
    flips = torch.rand(visible.shape, device=env.device) < visibility_flip_probability
    visible = torch.logical_xor(visible, flips) & active[:, 1:]
    slots = env._initial_obstacle_observation[ids].reshape(-1, 10, 4)
    capacity = objects.num_objects - 1
    slots[:, :capacity] = torch.where(visible[..., None], slots[:, :capacity], 0.0)
    slots[:, capacity:] = 0.0
    env._initial_obstacle_observation[ids] = slots.flatten(1)


def randomize_population_clutter_scene(
    env, env_ids, spawn_radius=0.02, goal_distance=0.15, object_radius=0.04,
    edge_clearance=0.02, shelf_half_extents=(0.20, 0.50), shelf_surface_height=1.05,
    excluded_x_half_angle=math.radians(30.0), observation_noise=0.01,
    visibility_threshold=0.35, visibility_flip_probability=0.01,
    gray_obstacles=1, white_obstacles=1,
):
    """Place exact active counts; layout failures never alter the sampled goal.

    At least the first Gray remains in the original behind-target region; any
    further Gray use the source Gray-zone rule. White stay outside Red/Gray.
    Inactive fixed slots are parked far above the scene with gravity disabled. Counts and
    masks change only for rows being reset, never midway through an episode.
    """
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    count = ids.numel()
    if not count:
        return
    values = (spawn_radius, goal_distance, object_radius, edge_clearance, shelf_surface_height,
              excluded_x_half_angle, observation_noise, *shelf_half_extents)
    if len(shelf_half_extents) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Scene sampling parameters must be finite and shelf_half_extents must have two axes")
    if observation_noise < 0 or edge_clearance < 0:
        raise ValueError("observation_noise and edge_clearance must be nonnegative")
    if not 0 <= excluded_x_half_angle < math.pi / 2:
        raise ValueError("excluded_x_half_angle must be in [0, pi/2)")
    if object_radius != 0.04 or tuple(shelf_half_extents) != (0.20, 0.50):
        raise ValueError("Scene geometry requires 4 cm cylinders on the 40 x 100 cm shelf")
    safe_half_extents = tuple(half - object_radius - edge_clearance for half in shelf_half_extents)
    if not (0 <= spawn_radius <= min(safe_half_extents) and goal_distance > 0):
        raise ValueError("Invalid target spawn radius or goal distance")
    max_gray, max_white = env.cfg.max_gray_obstacles, env.cfg.max_white_obstacles
    gray, white = min(gray_obstacles, max_gray), min(white_obstacles, max_white)
    active = active_population_mask(count, max_gray, max_white, gray, white, env.device)
    objects = env.scene["object_collection"]
    capacity = max_gray + max_white
    if objects.num_objects != capacity + 1 or objects.object_names != [
        "mass_cylinder", *(f"obstacle_{i}" for i in range(capacity))
    ]:
        raise ValueError("Population scene requires target index zero and fixed Gray-then-White slots")

    # Keep the exact parent samplers and call order. Never retry these samples.
    target_xy = _sample_uniform_disk(count, spawn_radius, env.device)
    direction_xy = _sample_planar_unit_vectors(count, env.device, excluded_x_half_angle)
    distance = _goal_ray_distance(target_xy, direction_xy, goal_distance, safe_half_extents)
    displacement_xy = direction_xy * distance[:, None]
    generator = population_layout_generator(env, gray, white)
    obstacle_cm = generator.generate(
        shelf_xy_to_generator_cm(target_xy.cpu().numpy()),
        shelf_xy_to_generator_cm((target_xy + displacement_xy).cpu().numpy()),
    )
    obstacle_xy = torch.as_tensor(generator_cm_to_shelf_xy(obstacle_cm), device=env.device, dtype=target_xy.dtype)
    local_positions = torch.zeros((count, capacity + 1, 3), device=env.device)
    local_positions[..., 2] = PARKING_HEIGHT
    local_positions[..., 1] = torch.arange(capacity + 1, device=env.device) * 0.2
    local_positions[:, 0, :2] = target_xy
    local_positions[:, 1:1 + gray, :2] = obstacle_xy[:, :gray]
    white_start = 1 + max_gray
    local_positions[:, white_start:white_start + white, :2] = obstacle_xy[:, gray:]
    local_positions[..., 2] = torch.where(active, shelf_surface_height, local_positions[..., 2])
    shelf = env.scene["shelf"].data
    shelf_rotation = shelf.root_quat_w[ids, None].expand(-1, capacity + 1, -1)
    world_positions = shelf.root_pos_w[ids, None] + quat_apply(shelf_rotation, local_positions)
    yaw = torch.empty((count, capacity + 1), device=env.device).uniform_(-math.pi, math.pi)
    yaw_rotation = torch.zeros((count, capacity + 1, 4), device=env.device)
    yaw_rotation[..., 0], yaw_rotation[..., 3] = (yaw / 2).cos(), (yaw / 2).sin()
    states = torch.cat((world_positions, quat_mul(shelf_rotation, yaw_rotation),
                        torch.zeros((count, capacity + 1, 6), device=env.device)), dim=-1)
    objects.write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = active
    apply_gravity_mask(env, ids)
    env.target_id[ids, 0] = 0
    env.target_width[ids, 0] = object_radius
    displacement = torch.cat((displacement_xy, torch.zeros((count, 1), device=env.device)), dim=-1)
    env.sweep_dir[ids] = quat_apply(shelf.root_quat_w[ids], displacement)

    robot = env.scene["robot"].data
    initial_positions, _ = subtract_frame_transforms(
        robot.root_pos_w[ids, None].expand(-1, capacity, -1),
        robot.root_quat_w[ids, None].expand(-1, capacity, -1), world_positions[:, 1:],
    )
    initial_positions += torch.empty_like(initial_positions).uniform_(-observation_noise, observation_noise)
    slots = torch.zeros((count, 10, 4), device=env.device)
    slots[:, :capacity, 0] = active[:, 1:]
    slots[:, :capacity, 1:] = torch.where(active[:, 1:, None], initial_positions, 0.0)
    env._initial_obstacle_observation[ids] = slots.flatten(1)
    apply_population_visibility(
        env, ids, object_radius=object_radius, visibility_threshold=visibility_threshold,
        visibility_flip_probability=visibility_flip_probability,
    )
