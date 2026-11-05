"""Initialize geometry/layouts/physics once; reset poses and observations only."""

import math
import torch
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms
from ..common.wrench_layout import LAYOUT_STAGES, WrenchLayoutBank
from .revision import initialize_revision_geometry
from .final import apply_final_visibility


def initialize_wrench_layouts(env, env_ids, stages=LAYOUT_STAGES, min_goal_distance=0.05):
    del env_ids
    params = env.cfg.events.object_spawn.params
    env._wrench_layout_bank = WrenchLayoutBank(
        env._revision_bounds_t, env._revision_layout_rng,
        size=env.cfg.layout_bank_size, batch_size=env.cfg.layout_build_batch_size,
        stages=stages, min_goal_distance=min_goal_distance,
        red_margin=env.cfg.red_margin, rear_offset=env.cfg.rear_offset,
        spawn_radius=params['spawn_radius'], goal_distance=params['goal_distance'],
        edge_clearance=params['edge_clearance'])
    env._wrench_gravity_flags = torch.ones((env.num_envs * 10, 1), dtype=torch.uint8)


def apply_wrench_gravity_mask(env, ids):
    """Transfer only reset rows; keep the full object-major PhysX buffer on CPU."""
    objects = env.scene['object_collection']
    cpu_ids = ids.cpu()
    rows = (torch.arange(objects.num_objects)[:, None] * env.num_envs + cpu_ids[None]).flatten()
    flags = (~env._active_object_mask[ids]).T.reshape(-1, 1).to(device='cpu', dtype=torch.uint8)
    env._wrench_gravity_flags[rows] = flags
    objects.root_physx_view.set_disable_gravities(env._wrench_gravity_flags, rows)
    active_rows = rows[flags[:, 0] == 0]
    if active_rows.numel():
        objects.root_physx_view.wake_up(active_rows)


def reset_wrench_scene(env, env_ids, gray_obstacles=1, white_obstacles=1,
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
    local, yaw, active, delta, actual, complete = env._wrench_layout_bank.sample(
        ids, env._revision_layout_rng, gray_obstacles, white_obstacles, excluded_x_half_angle)
    shelf = env.scene['shelf'].data
    rotation = shelf.root_quat_w[ids, None].expand(-1, 10, -1)
    world = shelf.root_pos_w[ids, None] + quat_apply(rotation, local)
    yaw_rotation = torch.zeros((len(ids), 10, 4), device=env.device)
    yaw_rotation[..., 0], yaw_rotation[..., 3] = (yaw / 2).cos(), (yaw / 2).sin()
    states = torch.cat((world, quat_mul(rotation, yaw_rotation),
                        torch.zeros((len(ids), 10, 6), device=env.device)), -1)
    env.scene['object_collection'].write_object_link_state_to_sim(states, env_ids=ids)
    env._active_object_mask[ids] = active
    apply_wrench_gravity_mask(env, ids)
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
    env._final_actual_population[ids] = actual
    env._revision_layout_complete[ids] = complete
    env._revision_spawn_positions[ids] = world
    env.sensor_data_fresh[ids] = False


def reset_wrench_feasibility_scene(env, env_ids, gray_obstacles=-1, white_obstacles=-1,
                                   spawn_radius=.02, goal_distance=.15, edge_clearance=.02,
                                   excluded_x_half_angle=math.pi / 6, observation_noise=.01,
                                   visibility_threshold=.35, visibility_flip_probability=.01,
                                   observation_swap_probability=.1, gray_padding=.065):
    """Draw independent integer offsets before clamping; reuse v0's bank/reset."""
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if not ids.numel():
        return
    counts = torch.randint(-1, 2, (len(ids), 2), device=env.device, generator=env._revision_layout_rng)
    counts[:, 0] = (counts[:, 0] + gray_obstacles).clamp(0, 4)
    counts[:, 1] = (counts[:, 1] + white_obstacles).clamp(0, 5)
    for gray, white in counts.unique(dim=0).tolist():
        selected = ids[(counts[:, 0] == gray) & (counts[:, 1] == white)]
        reset_wrench_scene(
            env, selected, gray_obstacles=gray, white_obstacles=white,
            spawn_radius=spawn_radius, goal_distance=goal_distance, edge_clearance=edge_clearance,
            excluded_x_half_angle=excluded_x_half_angle, observation_noise=observation_noise,
            visibility_threshold=visibility_threshold, visibility_flip_probability=visibility_flip_probability,
            observation_swap_probability=observation_swap_probability, gray_padding=gray_padding,
        )


def reset_wrench_uniform_scene(env, env_ids, spawn_radius=.02, goal_distance=.15,
                               edge_clearance=.02, excluded_x_half_angle=math.pi / 6,
                               observation_noise=.01, visibility_threshold=.35,
                               visibility_flip_probability=.01,
                               observation_swap_probability=.1, gray_padding=.065):
    """Independently sample Gray and White counts uniformly from 0 through 4."""
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if not ids.numel():
        return
    counts = torch.randint(0, 5, (len(ids), 2), device=env.device, generator=env._revision_layout_rng)
    for gray, white in counts.unique(dim=0).tolist():
        selected = ids[(counts[:, 0] == gray) & (counts[:, 1] == white)]
        reset_wrench_scene(
            env, selected, gray_obstacles=gray, white_obstacles=white,
            spawn_radius=spawn_radius, goal_distance=goal_distance, edge_clearance=edge_clearance,
            excluded_x_half_angle=excluded_x_half_angle, observation_noise=observation_noise,
            visibility_threshold=visibility_threshold, visibility_flip_probability=visibility_flip_probability,
            observation_swap_probability=observation_swap_probability, gray_padding=gray_padding,
        )
