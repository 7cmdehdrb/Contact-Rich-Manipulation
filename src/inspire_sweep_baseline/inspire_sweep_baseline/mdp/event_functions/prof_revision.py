"""Reference circle/Warp reset followed by one initial frontal OBB snapshot."""
from time import perf_counter
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate
from .wrench_refactor_relative_ik_test import fast_reset_scene
from .wrench_refactor import apply_fixed_pool_gravity_mask
from ..common.stabilize_sweep_layout import configure_test_layout_batch, test_inactive_positions
from ..common.stabilize_sweep_contact import cache_static_local_com
from ..common.prof_revision_obb_gpu import observe_batch, observe_hard_batch


def _snapshot(env, ids, local, yaw, active, shelf_state, *, hard=False):
    bounds = env._revision_bounds_t[ids]
    rotations = torch.zeros((*yaw.shape, 4), device=env.device)
    rotations[..., 0], rotations[..., 3] = (yaw/2).cos(), (yaw/2).sin()
    centers = local+quat_apply(rotations, bounds.mean(-2))
    boxes = torch.cat((centers, bounds[..., 1, :]-bounds[..., 0, :], yaw[..., None]), -1)
    observer = observe_hard_batch if hard else observe_batch
    target, obstacles, valid, source_slots, status = observer(
        boxes, active, env.cfg.obb_minimum_width, env.cfg.obb_minimum_ratio)
    target_missing, overflow = status == 1, status == 2
    failed = target_missing if hard else target_missing | overflow
    if bool((status == 3).any()):
        raise RuntimeError('Prof Revision observer active-source capacity exceeded')
    if hard and bool(target_missing.any()):
        raise RuntimeError(f'Hard layout occluded its target: envs={ids[target_missing].tolist()}')
    if bool(failed.any()):
        active[failed, 1:] = False
        fallback = observe_batch(boxes[failed], active[failed], env.cfg.obb_minimum_width, env.cfg.obb_minimum_ratio)
        if bool((fallback[-1] != 0).any()):
            raise RuntimeError(f'Prof Revision target-only OBB fallback failed: envs={ids[failed].tolist()}')
        for destination, source in zip((target, obstacles, valid, source_slots), fallback[:-1]):
            destination[failed] = source
    q = shelf_state[:, 3:7]
    def world(boxes):
        batched = boxes.ndim == 3
        quat = q[:, None].expand(-1, boxes.shape[1], -1) if batched else q
        pos = shelf_state[:, None, :3] if batched else shelf_state[:, :3]
        rotation = torch.zeros((*boxes.shape[:-1], 4), device=env.device)
        rotation[..., 0], rotation[..., 3] = (boxes[..., 6]/2).cos(), (boxes[..., 6]/2).sin()
        return pos+quat_apply(quat, boxes[..., :3]), quat_mul(quat, rotation)
    tp, tq = world(target)
    op, oq = world(obstacles)
    env._obb_target_pos_w[ids], env._obb_target_quat_w[ids] = tp, tq
    env._obb_target_size[ids] = target[:, 3:6]
    env._obb_obstacle_pos_w[ids], env._obb_obstacle_quat_w[ids] = op, oq
    env._obb_obstacle_size[ids], env._obb_valid[ids] = obstacles[..., 3:6], valid
    env._obb_sources[ids] = source_slots
    objects = env.scene['object_collection'].data
    inverse = quat_conjugate(objects.object_link_quat_w[ids, 0])
    env._obb_target_local_pos[ids] = quat_apply(inverse, tp-objects.object_link_pos_w[ids, 0])
    env._obb_target_local_quat[ids] = quat_mul(inverse, tq)
    return dict(fallbacks=failed.float().mean(), target_missing=target_missing.float().mean(),
                overflow=overflow.float().mean(), slots=valid.sum(-1).float().mean())


def reset_scene(env, env_ids, ik_params, spawn_radius=.02, edge_clearance=.02,
                excluded_x_half_angle=0., min_goal_distance=.06, max_goal_distance=.12,
                observation_noise=.01, visibility_threshold=.35,
                visibility_flip_probability=.01, observation_swap_probability=.1):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids):
        return
    start = perf_counter()
    hard = env.cfg.max_obstacles == 10
    if not hard:
        configure_test_layout_batch(env)
    parking = test_inactive_positions(env)[ids]
    fast_reset_scene(env, ids, ik_params=ik_params, spawn_radius=spawn_radius, edge_clearance=edge_clearance,
        excluded_x_half_angle=excluded_x_half_angle, min_goal_distance=min_goal_distance,
        max_goal_distance=max_goal_distance, observation_noise=observation_noise,
        visibility_threshold=visibility_threshold, visibility_flip_probability=visibility_flip_probability,
        observation_swap_probability=observation_swap_probability, reuse_unchanged_states=True,
        inactive_positions=parking, populate_legacy_observation=False)
    cache_static_local_com(env)
    placed = perf_counter()
    objects = env.scene['object_collection']
    shelf = env.scene['shelf'].data.root_state_w[ids]
    states = objects.data.object_link_state_w[ids].clone()
    q = shelf[:, None, 3:7].expand(-1, 17, -1)
    local = quat_apply_inverse(q, states[..., :3]-shelf[:, None, :3])
    relative = quat_mul(quat_conjugate(q), states[..., 3:7])
    w, x, y, z = relative.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    active = env._active_object_mask[ids].clone()
    requested = active[:, 1:].sum(-1)
    diagnostics = _snapshot(env, ids, local, yaw, active, shelf, hard=hard)
    changed = (active != env._active_object_mask[ids]).any(-1)
    if bool(changed.any()):
        local = torch.where(active[..., None], local, parking)
        states[..., :3] = shelf[:, None, :3]+quat_apply(q, local)
        # Match the reference target-only fallback orientation of retired slots.
        states[changed, 1:, 3:7] = q[changed, 1:]
        objects.write_object_link_state_to_sim(states[changed], env_ids=ids[changed])
        env._active_object_mask[ids] = active
        apply_fixed_pool_gravity_mask(env, ids[changed])
        env._revision_spawn_positions[ids] = states[..., :3]
    actual = active[:, 1:].sum(-1)
    env._final_actual_population[ids] = actual
    env._reset_diagnostics = {
        'Reset/requested_obstacles': requested.float().mean(),
        'Reset/actual_obstacles': actual.float().mean(),
        'Reset/empty_fraction': (actual == 0).float().mean(),
        'Reset/ik_attempts': 1., 'Reset/layout_resamples': 0.,
        'Reset/observation_fallbacks': diagnostics['fallbacks'],
        'Reset/target_missing_fallbacks': diagnostics['target_missing'],
        'Reset/slot_overflow_fallbacks': 0. if hard else diagnostics['overflow'],
        'Reset/obb_slots': diagnostics['slots'],
        'Reset/placement_ms': 1000*(placed-start),
        'Reset/observation_ms': 1000*(perf_counter()-placed),
        'Reset/batch_ms': 1000*(perf_counter()-start),
    }
    if hard:
        env._reset_diagnostics['Reset/slot_overflow_truncations'] = diagnostics['overflow']
        env._reset_diagnostics['Reset/layout_rank_retries'] = float(env._fast_layout.last_retries)


def reset_hard_scene(env, env_ids, ik_params, spawn_radius=.02, edge_clearance=.02,
                     excluded_x_half_angle=0., min_goal_distance=.05, max_goal_distance=None,
                     observation_noise=.01, visibility_threshold=.35,
                     visibility_flip_probability=.01, observation_swap_probability=.1):
    """Reuse the parent's physical/IK lifecycle with the ten-slot Hard layout."""
    return reset_scene(env, env_ids, ik_params, spawn_radius, edge_clearance,
                       excluded_x_half_angle, min_goal_distance, max_goal_distance,
                       observation_noise, visibility_threshold, visibility_flip_probability,
                       observation_swap_probability)
