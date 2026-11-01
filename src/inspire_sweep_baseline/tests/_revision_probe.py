"""Real Isaac Lab initialization, reset, sensor, OBB and rollout checks."""

import argparse
import faulthandler
import json
from pathlib import Path
import re
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--resets', type=int, default=4)
parser.add_argument('--output', type=Path, default=Path('outputs/revision_validation'))
parser.add_argument('--bbox', action='store_true')
parser.add_argument('--scale-endpoints', action='store_true',
                    help='Test exact 0.9/1.1 instance scales; preserve random shape selection.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
faulthandler.dump_traceback_later(120, repeat=True)
env = None
raw = None

try:
    import numpy as np
    import torch
    import gymnasium as gym
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim
    import inspire_sweep_baseline
    from inspire_sweep_baseline.envs.revision_envs import RevisionEnvCfg
    from inspire_sweep_baseline.mdp.common.bbox_scene_layout import axes, footprint_from_bounds, separated
    from inspire_sweep_baseline.mdp.common.revision_goal import object_geometry_in_shelf
    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

    started = time.perf_counter()
    cfg = RevisionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.target_bounding_box = args.bbox
    cfg.log_dir = str(args.output)
    if args.scale_endpoints:
        from inspire_sweep_baseline.assets import revision_spawners

        # Replace only this module's namespace while constructing the scene.
        # The process-wide random module and random shape selector stay intact.
        original_random = revision_spawners.random
        endpoint_draw_count = 0

        def endpoint_uniform(low, high):
            global endpoint_draw_count
            assert (low, high) == (.9, 1.1), (low, high)
            value = low if endpoint_draw_count % 2 == 0 else high
            endpoint_draw_count += 1
            return value

        revision_spawners.random = SimpleNamespace(randrange=original_random.randrange, uniform=endpoint_uniform)
        try:
            env = gym.make('Isaac-Inspire-Revision-v0', cfg=cfg)
        finally:
            revision_spawners.random = original_random
    else:
        env = gym.make('Isaac-Inspire-Revision-v0', cfg=cfg)
    raw = env.unwrapped
    initialized_s = time.perf_counter() - started
    obs, _ = env.reset(seed=42)
    from revision_contact_geometry_probe import measure_contact_height_geometry
    contact_geometry = measure_contact_height_geometry(raw)
    assert all(contact_geometry['side_push']['object_specific_feasible']), contact_geometry['side_push']
    assert abs(contact_geometry['hand_current_minimum_floor_clearance_m']) < 1.
    print('REVISION_CONTACT_GEOMETRY ' + json.dumps(contact_geometry), flush=True)
    expected = {'policy': 57, 'obstacles': 64, 'privileged': 3, 'critic_obstacles': 64}
    assert {key: value.shape[-1] for key, value in obs.items()} == expected
    assert raw.scene['object_collection'].num_objects == 10
    assert len(raw.cfg.scene.object_collection.rigid_objects) == 10
    assert np.all(raw._final_object_scales >= .9) and np.all(raw._final_object_scales <= 1.1)
    stage = sim.get_current_stage()
    bbox_paths = [str(p.GetPath()) for p in stage.Traverse() if p.GetName() == 'TargetBoundingBox']
    assert len(bbox_paths) == (args.num_envs if args.bbox else 0), bbox_paths[:10]
    assert all('/mass_cylinder/' in p for p in bbox_paths)
    objects = raw.scene['object_collection']
    # The collection view is a simulator count; the USD audit additionally
    # catches nested enabled bodies and forgotten on-stage template objects.
    assert objects.root_physx_view.count == args.num_envs * 10
    assert not stage.GetPrimAtPath('/World/RevisionObjectTemplates')
    object_body_paths, other_world_bodies = [], []
    usd_body_count = 0
    object_path_pattern = re.compile(r'^/World/envs/env_\d+/(mass_cylinder|obstacle_\d+)(?:/|$)')
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is False:
            continue
        usd_body_count += 1
        path = str(prim.GetPath())
        if object_path_pattern.match(path):
            object_body_paths.append(path)
        elif not path.startswith('/World/envs/env_'):
            other_world_bodies.append(path)
    assert len(object_body_paths) == args.num_envs * 10, len(object_body_paths)
    assert not other_world_bodies, other_world_bodies[:10]

    # Check one available instance of each kind, selecting both scale extremes
    # when available. Query enabled collider bounds including invisible YCB
    # proxies; visual-only target boxes must never enter this measurement.
    kinds = np.asarray(raw._revision_object_kinds)
    scales = np.asarray(raw._final_object_scales)
    baseline_bounds = np.asarray(raw._revision_base_object_bounds)
    endpoint_spawn_coverage = {
        str(endpoint): np.bincount(kinds[scales == endpoint], minlength=8).tolist()
        for endpoint in (.9, 1.1)
    }
    if args.scale_endpoints:
        assert np.all((scales == .9) | (scales == 1.1))
        assert endpoint_draw_count == args.num_envs * 10
        assert all(all(count > 0 for count in values) for values in endpoint_spawn_coverage.values()), (
            'Increase --num_envs to cover all eight kinds at both exact scale endpoints', endpoint_spawn_coverage)
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'],
                                    useExtentsHint=False, ignoreVisibility=True)
    collider_checks = []
    for kind in range(8):
        available = np.argwhere(kinds == kind)
        if not len(available):
            continue
        values = scales[available[:, 0], available[:, 1]]
        selected = {tuple(available[int(values.argmin())]), tuple(available[int(values.argmax())])}
        for row, slot in sorted(selected):
            name = objects.object_names[slot]
            root = stage.GetPrimAtPath(f'/World/envs/env_{row}/{name}')
            scale = float(scales[row, slot])
            root_scales = [op.Get() for op in UsdGeom.Xformable(root).GetOrderedXformOps()
                           if op.GetOpType() == UsdGeom.XformOp.TypeScale]
            assert len(root_scales) == 1
            np.testing.assert_allclose(tuple(root_scales[0]), (scale,) * 3, atol=1e-7, rtol=0)
            placement_bounds = np.asarray(raw._final_object_bounds[row, slot])
            np.testing.assert_allclose(placement_bounds, baseline_bounds[kind] * scale, atol=1e-7, rtol=0)
            collision_range = Gf.Range3d()
            collider_count = 0
            for child in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not child.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                if UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is False:
                    continue
                collision_range.UnionWith(bounds_cache.ComputeRelativeBound(child, root).ComputeAlignedRange())
                collider_count += 1
            assert collider_count > 0, name
            collider_bounds = np.asarray([tuple(collision_range.GetMin()), tuple(collision_range.GetMax())]) * scale
            # Mesh proxies are contained by the planning box; submillimetre
            # USD float quantization is tolerated, not a separate spawn offset.
            assert np.all(collider_bounds[0] >= placement_bounds[0] - 2e-4), (kind, collider_bounds, placement_bounds)
            assert np.all(collider_bounds[1] <= placement_bounds[1] + 2e-4), (kind, collider_bounds, placement_bounds)
            assert abs(collider_bounds[0, 2] - placement_bounds[0, 2]) <= 2e-4, (kind, collider_bounds, placement_bounds)
            collider_checks.append({'kind': int(kind), 'env': int(row), 'slot': int(slot),
                                    'scale': scale, 'enabled_colliders': collider_count,
                                    'collider_bounds': collider_bounds.tolist(),
                                    'placement_bounds': placement_bounds.tolist()})

    def assert_zero_padding(observation):
        actor = observation['obstacles'].reshape(args.num_envs, 16, 4)
        critic = observation['critic_obstacles'].reshape(args.num_envs, 16, 4)
        assert (actor[actor[..., 0] == 0] == 0).all()
        assert (critic[critic[..., 0] == 0] == 0).all()
        torch.testing.assert_close(critic[..., 0].sum(-1), raw._active_object_mask[:, 1:].sum(-1).float())

    def assert_reset_geometry(env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        shelf = raw.scene['shelf'].data
        sq = shelf.root_quat_w[selected, None].expand(-1, 10, -1)
        local = quat_apply_inverse(sq, objects.data.object_pos_w[selected] - shelf.root_pos_w[selected, None])
        rotation = quat_mul(quat_conjugate(sq), objects.data.object_quat_w[selected])
        # Every reset is upright; derive yaw from actual simulator quaternions.
        yaw = torch.atan2(2 * (rotation[..., 0] * rotation[..., 3] + rotation[..., 1] * rotation[..., 2]),
                          1 - 2 * (rotation[..., 2].square() + rotation[..., 3].square()))
        active = raw._active_object_mask[selected]
        half, offset = footprint_from_bounds(raw._revision_bounds_t[selected], yaw)
        centers = local[..., :2] + offset
        signs = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], device=raw.device)
        corners = torch.einsum('bmki,bmij->bmkj', half[:, :, None] * signs, axes(yaw)) + centers[:, :, None]
        # Five-millimetre wall margin and twenty-millimetre SAT gap are the
        # requested layout contract. 20um only covers tensor/PhysX round trips.
        assert (corners[..., 0].abs()[active] <= .19502).all()
        assert (corners[..., 1].abs()[active] <= .49502).all()
        for left in range(10):
            for right in range(left + 1, 10):
                both = active[:, left] & active[:, right]
                assert separated(centers[:, left], half[:, left], yaw[:, left],
                                 centers[:, right], half[:, right], yaw[:, right], .02 - 2e-5)[both].all(), (left, right)
        bottom = object_geometry_in_shelf(raw)[1][selected]
        torch.testing.assert_close(bottom[active], torch.full_like(bottom[active], 1.05), atol=2e-5, rtol=0)

    counts = []
    for iteration in (0, 2000, 2500, 5000, 10000):
        raw.set_learning_iteration(iteration)
        obs, _ = env.reset()
        assert raw.cfg.goal_success_radius == .03
        assert_zero_padding(obs)
        assert_reset_geometry()
        active_np = raw._active_object_mask.cpu().numpy()
        counts.append({'iteration': iteration, 'requested': [raw.event_manager.get_term_cfg('object_spawn').params['gray_obstacles'], raw.event_manager.get_term_cfg('object_spawn').params['white_obstacles']],
                       'actual_mean': raw._final_actual_population.float().mean(0).tolist(),
                       'active_kind_counts': {
                           'target': np.bincount(kinds[:, 0][active_np[:, 0]], minlength=8).tolist(),
                           'gray': np.bincount(kinds[:, 1:5][active_np[:, 1:5]], minlength=8).tolist(),
                           'white': np.bincount(kinds[:, 5:10][active_np[:, 5:10]], minlength=8).tolist(),
                       }})

    # Capture post-physics data before ManagerBasedRLEnv automatically resets
    # terminated rows. Checking only env.step() outputs would hide a launch.
    captured = {}
    original_compute = raw.termination_manager.compute

    def compute_and_capture():
        result = original_compute()
        captured['position'] = objects.data.object_pos_w.clone()
        captured['velocity'] = objects.data.object_lin_vel_w.clone()
        captured['active'] = raw._active_object_mask.clone()
        captured['bottom'] = object_geometry_in_shelf(raw)[1].clone()
        captured['causes'] = {name: int(raw.termination_manager.get_term(name).sum())
                              for name in raw.termination_manager.active_terms}
        return result

    raw.termination_manager.compute = compute_and_capture
    action = torch.zeros((args.num_envs, raw.action_manager.total_action_dim), device=raw.device)
    action[:, -2:] = 1.

    def stationary_phase(name, iteration, steps):
        raw.set_learning_iteration(iteration)
        obs, _ = env.reset()
        assert_zero_padding(obs)
        assert_reset_geometry()
        actual_counts = raw._final_actual_population.float().mean(0).tolist()
        initial_active = raw._active_object_mask.cpu().numpy()
        role_kinds = {
            'target': np.bincount(kinds[:, 0][initial_active[:, 0]], minlength=8).tolist(),
            'gray': np.bincount(kinds[:, 1:5][initial_active[:, 1:5]], minlength=8).tolist(),
            'white': np.bincount(kinds[:, 5:10][initial_active[:, 5:10]], minlength=8).tolist(),
        }
        active_endpoint_kinds = {
            str(endpoint): np.bincount(kinds[initial_active & (scales == endpoint)], minlength=8).tolist()
            for endpoint in (.9, 1.1)
        }
        start_positions = objects.data.object_pos_w.clone()
        max_speed, max_drift, inactive_max_drift, inactive_max_speed = 0., 0., 0., 0.
        min_bottom, early_max_speed, early_max_drift = float('inf'), 0., 0.
        automatic_resets, inactive_samples_count = 0, 0
        reset_causes = {term: 0 for term in raw.termination_manager.active_terms}
        for step in range(steps):
            obs, reward, terminated, truncated, info = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert_zero_padding(obs)
            active = captured['active']
            speed = captured['velocity'].norm(dim=-1)
            displacement = (captured['position'] - start_positions).norm(dim=-1)
            assert torch.isfinite(speed).all() and torch.isfinite(displacement).all()
            max_speed = max(max_speed, float(speed[active].max()))
            max_drift = max(max_drift, float(displacement[active].max()))
            min_bottom = min(min_bottom, float(captured['bottom'][active].min()))
            if step < 5:
                early_max_speed = max(early_max_speed, float(speed[active].max()))
                early_max_drift = max(early_max_drift, float(displacement[active].max()))
            if (~active).any():
                inactive_samples_count += int((~active).sum())
                inactive_max_drift = max(inactive_max_drift, float(displacement[~active].max()))
                inactive_max_speed = max(inactive_max_speed, float(speed[~active].max()))
            # Stationary arm/open hand: allow settling, but reject the same
            # 1m/s speed and 1cm bottom-drop limits used by task failures. A
            # 5cm travel allowance is much larger than the 1mm contact offset,
            # yet catches thrown objects before automatic episode resets.
            assert max_speed <= 1.0 + 1e-4, f'{name} launch speed: {max_speed}'
            assert max_drift <= .05, f'{name} stationary displacement: {max_drift}'
            assert min_bottom >= 1.04 - 1e-4, f'{name} object bottom: {min_bottom}'
            assert inactive_max_drift <= 1e-4, f'{name} inactive drift: {inactive_max_drift}'
            assert inactive_max_speed <= 1e-3, f'{name} inactive velocity: {inactive_max_speed}'
            reset = terminated | truncated
            automatic_resets += int(reset.sum())
            for term, value in captured['causes'].items():
                reset_causes[term] += value
            # Start the next episode only after auditing the prior terminal
            # state, preventing both hidden launches and false reset drifts.
            start_positions[reset] = objects.data.object_pos_w[reset]
        if name == 'sparse':
            assert inactive_samples_count > 0, 'Sparse phase must test real parked bodies'
        return {
            'phase': name, 'iteration': iteration, 'policy_steps': steps,
            'physics_steps': steps * raw.cfg.decimation,
            'requested': [raw.event_manager.get_term_cfg('object_spawn').params['gray_obstacles'],
                          raw.event_manager.get_term_cfg('object_spawn').params['white_obstacles']],
            'actual_mean': actual_counts, 'active_kind_counts': role_kinds,
            'active_endpoint_kind_counts': active_endpoint_kinds,
            'max_active_speed': max_speed, 'max_active_displacement': max_drift,
            'minimum_active_bottom': min_bottom,
            'early_first_five_policy_steps': {'max_speed': early_max_speed, 'max_displacement': early_max_drift},
            'inactive_samples_count': inactive_samples_count,
            'inactive_max_displacement': inactive_max_drift, 'inactive_max_speed': inactive_max_speed,
            'automatic_resets': automatic_resets, 'automatic_reset_causes': reset_causes,
        }

    # A final dense scene may contain no inactive body. Test sparse episodes
    # explicitly, then the dense population for at least thirty policy steps.
    phase_results = [stationary_phase('sparse', 0, 5),
                     stationary_phase('dense', 10000, max(30, args.steps))]
    endpoint_active_coverage = {
        str(endpoint): np.sum([phase['active_endpoint_kind_counts'][str(endpoint)]
                              for phase in phase_results], axis=0).tolist()
        for endpoint in (.9, 1.1)
    }
    if args.scale_endpoints:
        assert all(all(count > 0 for count in values) for values in endpoint_active_coverage.values()), (
            'Every kind/scale endpoint must participate in actual stationary physics', endpoint_active_coverage)
        audited = {(entry['kind'], entry['scale']) for entry in collider_checks}
        assert {(kind, endpoint) for kind in range(8) for endpoint in (.9, 1.1)} <= audited
    max_speed = max(phase['max_active_speed'] for phase in phase_results)
    max_drift = max(phase['max_active_displacement'] for phase in phase_results)
    min_bottom = min(phase['minimum_active_bottom'] for phase in phase_results)
    inactive_max_drift = max(phase['inactive_max_displacement'] for phase in phase_results)
    inactive_max_speed = max(phase['inactive_max_speed'] for phase in phase_results)
    inactive_samples_count = sum(phase['inactive_samples_count'] for phase in phase_results)
    assert inactive_samples_count > 0
    automatic_resets = sum(phase['automatic_resets'] for phase in phase_results)
    reset_causes = {term: sum(phase['automatic_reset_causes'][term] for phase in phase_results)
                    for term in raw.termination_manager.active_terms}
    raw.termination_manager.compute = original_compute
    ids = torch.arange(0, args.num_envs, 2, device=raw.device)
    other = torch.arange(1, args.num_envs, 2, device=raw.device)
    for _ in range(args.resets):
        before = objects.data.object_link_state_w[other].clone()
        obs, _ = raw.reset(env_ids=ids)
        torch.testing.assert_close(objects.data.object_link_state_w[other], before, atol=1e-6, rtol=0)
        assert_zero_padding(obs)
        assert_reset_geometry(ids)
    heights = baseline_bounds[:, 1, 2] - baseline_bounds[:, 0, 2]
    report = {'task': 'Isaac-Inspire-Revision-v0', 'envs': args.num_envs,
              'contact_geometry': contact_geometry,
              'physical_objects_per_env': 10, 'observation_dimensions': expected,
              'initialization_s': initialized_s, 'total_s': time.perf_counter() - started,
              'steps': sum(phase['policy_steps'] for phase in phase_results),
              'requested_dense_steps': args.steps, 'partial_resets': args.resets, 'counts': counts,
              'stationary_phases': phase_results,
              'scale_endpoint_mode': args.scale_endpoints,
              'endpoint_spawn_kind_counts': endpoint_spawn_coverage,
              'endpoint_active_kind_counts': endpoint_active_coverage,
              'max_active_speed': max_speed, 'max_active_displacement': max_drift,
              'minimum_active_bottom': min_bottom,
              'inactive_samples_count': inactive_samples_count,
              'inactive_max_displacement': inactive_max_drift, 'inactive_max_speed': inactive_max_speed,
              'automatic_resets': automatic_resets, 'automatic_reset_causes': reset_causes,
              'usd_enabled_rigid_bodies': usd_body_count, 'usd_object_bodies': len(object_body_paths),
              'physx_object_bodies': objects.root_physx_view.count,
              'extra_world_rigid_bodies': other_world_bodies,
              'collider_metadata_checks': collider_checks,
              'baseline_kind_names': list(raw._revision_object_kind_names),
              'baseline_shape_heights': heights.tolist(), 'baseline_mean_height': float(heights.mean()),
              'contact_height_limit_above_initial_surface': raw.cfg.contact_height_limit,
              'kind_counts': np.bincount(np.asarray(raw._revision_object_kinds).reshape(-1), minlength=8).tolist(),
              'scale_min': float(np.min(raw._final_object_scales)), 'scale_max': float(np.max(raw._final_object_scales)),
              'torch_peak_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if str(raw.device) != 'cpu' else None,
              'torch_peak_reserved_mb': torch.cuda.max_memory_reserved() / 2**20 if str(raw.device) != 'cpu' else None,
              'gpu_free_total_mb': [value / 2**20 for value in torch.cuda.mem_get_info()] if str(raw.device) != 'cpu' else None,
              'bbox_count': len(bbox_paths)}
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = '_endpoints' if args.scale_endpoints else ''
    (args.output / f'probe_{args.num_envs}{suffix}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REVISION_PROBE ' + json.dumps(report), flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    elif raw is not None:
        raw.close()
    app.close()
