"""Sample the inherited target/goal first, then place obstacles around that goal."""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from ..mdp.goal_events import _goal_ray_distance, _sample_planar_unit_vectors, _sample_uniform_disk
from .layout import DEFAULT_LAYOUT_CONFIG, ClutterLayoutGenerator, generator_cm_to_shelf_xy, shelf_xy_to_generator_cm


def randomize_clutter_scene(
    env, env_ids, spawn_radius=0.02, goal_distance=0.15, object_radius=0.04,
    edge_clearance=0.02, shelf_half_extents=(0.20, 0.50), shelf_surface_height=1.05,
    excluded_x_half_angle=math.radians(15.0), observation_noise=0.01,
):
    """Retain parent distributions; a placement failure never resamples the goal."""
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    count = ids.numel()
    if count == 0:
        return
    values = (
        spawn_radius, goal_distance, object_radius, edge_clearance,
        shelf_surface_height, excluded_x_half_angle, observation_noise, *shelf_half_extents,
    )
    if len(shelf_half_extents) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Scene sampling parameters must be finite and shelf_half_extents must have two axes")
    if observation_noise < 0.0 or edge_clearance < 0.0:
        raise ValueError("observation_noise and edge_clearance must be nonnegative")
    if not 0.0 <= excluded_x_half_angle < math.pi / 2:
        raise ValueError("excluded_x_half_angle must be in [0, pi/2)")
    objects = env.scene["object_collection"]
    obstacle_count = DEFAULT_LAYOUT_CONFIG.obstacles
    object_count = obstacle_count + 1
    if objects.num_objects != object_count or objects.object_names[0] != "mass_cylinder":
        raise ValueError(f"Clutter scene requires target index 0 followed by {obstacle_count} obstacles")
    if object_radius != 0.04 or tuple(shelf_half_extents) != (0.20, 0.50):
        raise ValueError("Scene randomization geometry requires 4 cm cylinders on the 40 x 100 cm shelf")
    safe_half_extents = tuple(half - object_radius - edge_clearance for half in shelf_half_extents)
    if not (0 <= spawn_radius <= min(safe_half_extents) and goal_distance > 0):
        raise ValueError("Invalid target spawn radius or goal distance")

    # These are exactly the parent's target/command samplers and clipping rule.
    target_xy = _sample_uniform_disk(count, spawn_radius, env.device)
    direction_xy = _sample_planar_unit_vectors(count, env.device, excluded_x_half_angle)
    distance = _goal_ray_distance(target_xy, direction_xy, goal_distance, safe_half_extents)
    displacement_xy = direction_xy * distance[:, None]
    goal_xy = target_xy + displacement_xy
    if not hasattr(env, "_clutter_layout_generator"):
        env._clutter_layout_generator = ClutterLayoutGenerator(seed=env.cfg.seed)
    obstacle_cm = env._clutter_layout_generator.generate(
        shelf_xy_to_generator_cm(target_xy.cpu().numpy()),
        shelf_xy_to_generator_cm(goal_xy.cpu().numpy()),
    )
    obstacle_xy = torch.as_tensor(generator_cm_to_shelf_xy(obstacle_cm), device=env.device, dtype=target_xy.dtype)
    local_positions = torch.zeros((count, object_count, 3), device=env.device)
    local_positions[:, 0, :2] = target_xy
    local_positions[:, 1:, :2] = obstacle_xy
    local_positions[..., 2] = shelf_surface_height
    shelf = env.scene["shelf"].data
    shelf_rotation = shelf.root_quat_w[ids, None, :].expand(-1, object_count, -1)
    world_positions = shelf.root_pos_w[ids, None, :] + quat_apply(shelf_rotation, local_positions)
    yaw = torch.empty((count, object_count), device=env.device).uniform_(-math.pi, math.pi)
    yaw_rotation = torch.zeros((count, object_count, 4), device=env.device)
    yaw_rotation[..., 0] = (yaw / 2).cos()
    yaw_rotation[..., 3] = (yaw / 2).sin()
    states = torch.cat((
        world_positions, quat_mul(shelf_rotation, yaw_rotation),
        torch.zeros((count, object_count, 6), device=env.device),
    ), dim=-1)
    objects.write_object_link_state_to_sim(states, env_ids=ids)
    env.target_id[ids, 0] = 0
    env.target_width[ids, 0] = object_radius
    displacement = torch.cat((displacement_xy, torch.zeros((count, 1), device=env.device)), dim=-1)
    env.sweep_dir[ids] = quat_apply(shelf.root_quat_w[ids], displacement)

    robot = env.scene["robot"].data
    initial_root_positions, _ = subtract_frame_transforms(
        robot.root_pos_w[ids, None, :].expand(-1, obstacle_count, -1),
        robot.root_quat_w[ids, None, :].expand(-1, obstacle_count, -1),
        world_positions[:, 1:],
    )
    initial_root_positions += torch.empty_like(initial_root_positions).uniform_(
        -observation_noise, observation_noise,
    )
    slots = torch.zeros((count, 10, 4), device=env.device)
    slots[:, :obstacle_count, 0] = 1.0
    slots[:, :obstacle_count, 1:] = initial_root_positions
    env._initial_obstacle_observation[ids] = slots.flatten(1)
