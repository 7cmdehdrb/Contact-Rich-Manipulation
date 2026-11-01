"""Revision lifecycle events, independent of the Final environment variants."""

import math
import torch
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from ..common.revision_layout import revision_layout
from .contact_sweep_clutter_fusion import apply_gravity_mask
from .final import apply_final_visibility


def initialize_revision_geometry(env, env_ids=None):
    from ...assets.revision_spawners import collect_revision_object_metadata
    collect_revision_object_metadata(env, env_ids)
    env._revision_bounds_t = torch.as_tensor(env._final_object_bounds, device=env.device, dtype=torch.float32)
    if env.cfg.target_bounding_box:
        from ...assets.final_visualization import create_target_bounding_boxes
        create_target_bounding_boxes(env)


def reset_revision_scene(env, env_ids, gray_obstacles=1, white_obstacles=1,
                         spawn_radius=.02, goal_distance=.15, edge_clearance=.02,
                         excluded_x_half_angle=math.pi / 6, observation_noise=.01,
                         visibility_threshold=.35, visibility_flip_probability=.01,
                         observation_swap_probability=.1, gray_padding=.065):
    if gray_padding != .065:
        raise ValueError('Revision Gray padding is fixed at 65 mm')
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if not ids.numel():
        return
    local, yaw, active, delta, result = revision_layout(
        env._revision_bounds_t[ids], env._revision_layout_rng,
        gray_obstacles, white_obstacles, spawn_radius=spawn_radius,
        goal_distance=goal_distance, edge_clearance=edge_clearance,
        excluded_x_half_angle=excluded_x_half_angle)
    shelf = env.scene['shelf'].data
    rotation = shelf.root_quat_w[ids, None].expand(-1, 10, -1)
    world = shelf.root_pos_w[ids, None] + quat_apply(rotation, local)
    yaw_rotation = torch.zeros((len(ids), 10, 4), device=env.device)
    yaw_rotation[..., 0], yaw_rotation[..., 3] = (yaw / 2).cos(), (yaw / 2).sin()
    states = torch.cat((world, quat_mul(rotation, yaw_rotation),
                        torch.zeros((len(ids), 10, 6), device=env.device)), -1)
    env.scene['object_collection'].write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = active
    apply_gravity_mask(env, ids)
    env.target_id[ids, 0] = 0
    half = env._revision_bounds_t[ids, 0, :, :2].abs().amax(1)
    env.target_width[ids, 0] = torch.linalg.vector_norm(half, dim=-1)
    direction = torch.zeros((len(ids), 3), device=env.device)
    direction[:, :2] = delta
    env.sweep_dir[ids] = quat_apply(shelf.root_quat_w[ids], direction)
    robot = env.scene['robot'].data
    positions, _ = subtract_frame_transforms(robot.root_pos_w[ids, None].expand(-1, 9, -1),
        robot.root_quat_w[ids, None].expand(-1, 9, -1), world[:, 1:])
    positions += torch.rand(positions.shape, device=env.device, generator=env._revision_layout_rng) * (2 * observation_noise) - observation_noise
    slots = torch.zeros((len(ids), 16, 4), device=env.device)
    slots[:, :9, 0] = active[:, 1:]
    slots[:, :9, 1:] = torch.where(active[:, 1:, None], positions, 0.)
    env._initial_obstacle_observation[ids] = slots.flatten(1)
    apply_final_visibility(env, ids, visibility_threshold, visibility_flip_probability, observation_swap_probability)
    env._final_actual_population[ids] = result.actual
    env._revision_layout_complete[ids] = result.complete
    env._revision_spawn_positions[ids] = world
    env.sensor_data_fresh[ids] = False
