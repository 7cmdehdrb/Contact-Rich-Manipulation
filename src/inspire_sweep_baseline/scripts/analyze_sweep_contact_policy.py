"""Read-only policy rollout: pre-reset contacts, trajectories and palm orientation.

Run with Isaac Lab Python. No environment, reward, checkpoint or training-log edits.
All vectors/points are saved in the initial shelf frame; plotting uses screen
right=-shelf Y, up=+shelf X, matching record_three_views inside_top_down.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--task', default='Inspire-Sweep-Only-v2-Test-ActionPenalty-LinearReach')
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--surface-diagnostics', action='store_true',
                        help='Record reward-side palm/dorsal loads and physical failure diagnostics.')
    parser.add_argument('--reward-timeline', action='store_true',
                        help='Record exact raw/weighted/dt-applied reward terms and their state inputs.')
    parser.add_argument('--initial-direction-center', type=float,
                        help='Rejection-sample only initial scenes in this top-view sector (degrees clockwise from up).')
    parser.add_argument('--initial-direction-half-width', type=float, default=15.)
    parser.add_argument('--output-dir', type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.reward_timeline:
        args.surface_diagnostics = True
    if args.num_envs < 1 or args.steps < 1:
        parser.error('Positive environments and steps required')
    checkpoint = args.checkpoint.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from cli_args import apply_renderer_compatibility
    apply_renderer_compatibility(args)
    app = AppLauncher(args).app
    env = None
    report = dict(status='initializing', task=args.task, checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  seed=args.seed, num_envs=args.num_envs, requested_steps=args.steps,
                  coordinate_frame='initial shelf', screen_axes='right=-Y, up=+X',
                  timing='after termination compute, before reward/reset',
                  policy='deterministic inference; default environment observation noise retained')
    try:
        import numpy as np
        import torch
        import gymnasium as gym
        import inspire_sweep_baseline  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        from rsl_rl.runners import OnPolicyRunner
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        from inspire_sweep_baseline.mdp.common.stabilize_sweep_contact import centroid
        from inspire_sweep_baseline.assets.robot import ARM_JOINT_NAMES, HAND_CONTACT_BODY_NAMES, HAND_LINK_BODY_NAMES
        from inspire_sweep_baseline.assets.hand_kinematics import HAND_DRIVE_JOINT_NAMES

        cfg = parse_env_cfg(args.task, device=args.device or 'cuda:0', num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.log_dir = str(args.output_dir.resolve())
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = min(
            cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity, 2**21)
        agent = load_cfg_from_registry(args.task, 'rsl_rl_cfg_entry_point')
        agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
        agent.device = cfg.sim.device
        agent.seed = args.seed
        runner_type = {'OnPolicyRunner': OnPolicyRunner, 'CurriculumOnPolicyRunner': CurriculumOnPolicyRunner}[agent.class_name]
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
        runner = runner_type(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        env.reset()
        raw = env.unwrapped
        n = raw.num_envs
        origins = raw._revision_initial_shelf_pos.clone()
        rotation = raw._revision_initial_shelf_quat.clone()
        def vector(value):
            return quat_apply_inverse(rotation, value)
        def point(value):
            return vector(value - origins)
        if args.initial_direction_center is not None:
            attempts = 0
            while True:
                delta = vector(raw.command_manager.get_command('target_goal_pos')[:, :3] - centroid(raw))
                angle = torch.rad2deg(torch.atan2(-delta[:, 1], delta[:, 0]))
                gap = (angle - args.initial_direction_center + 180) % 360 - 180
                rejected = (gap.abs() > args.initial_direction_half_width).nonzero(as_tuple=False).flatten()
                if not len(rejected):
                    break
                if attempts >= 200:
                    raise RuntimeError('Initial direction rejection sampling did not converge')
                raw.reset(env_ids=rejected)
                attempts += 1
            report.update(initial_direction_center=args.initial_direction_center,
                          initial_direction_half_width=args.initial_direction_half_width,
                          initial_rejection_rounds=attempts,
                          conditional_initial_distribution=True,
                          conditional_outcomes='Use only episode=0 per environment; later auto-resets are unconditioned')
            print(f'[Analysis] Initial direction sector accepted after {attempts} partial-reset rounds', flush=True)
        initial_target = point(centroid(raw)).clone()
        initial_goal = point(raw.command_manager.get_command('target_goal_pos')[:, :3]).clone()
        episode = torch.zeros(n, dtype=torch.long, device=raw.device)
        previous_distance = (initial_target[:, :2] - initial_goal[:, :2]).norm(dim=-1)
        all_rows = {}
        substeps = []
        max_count = 0
        force_error = 0.0
        contact = raw._stabilize_contacts
        original_capture = contact.capture
        slots = torch.arange(64, device=raw.device)[None, None, :]
        hand_ids = [raw.scene['robot'].joint_names.index(name) for name in HAND_DRIVE_JOINT_NAMES]
        body_ids = [raw.scene['robot'].body_names.index(name) for name in HAND_LINK_BODY_NAMES]
        arm_ids = [raw.scene['robot'].joint_names.index(name) for name in ARM_JOINT_NAMES]

        def capture(source):
            nonlocal max_count, force_error
            original_capture(source)
            # Normal data and force matrix are queried at the same physics state.
            scalar, positions, normals, _, counts, starts = contact.view.get_contact_data(dt=raw.physics_dt)
            max_count = max(max_count, int(counts.max().item()))
            if max_count > 64:
                raise RuntimeError('Per-filter contact point count exceeds analysis slots')
            indices = starts[..., None] + slots
            mask = (slots < counts[..., None]) & (indices >= 0) & (indices < len(positions))
            indices = indices.clamp(0, len(positions) - 1).long()
            weights = scalar.reshape(-1)[indices].abs() * mask
            pair_load = weights.sum(-1)
            load = pair_load.sum(-1)
            weighted = (positions[indices] * weights[..., None]).sum((1, 2))
            normal = (scalar.reshape(-1)[indices][..., None] * normals[indices] * mask[..., None]).sum((1, 2))
            matrix_normal = contact.view.get_contact_force_matrix(dt=raw.physics_dt).sum(1).clone()
            force_error = max(force_error, float((normal - matrix_normal).abs().max().item()))
            friction = contact.force - normal
            substeps.append((normal.clone(), friction.clone(), weighted.clone(), load.clone(), pair_load.clone()))

        contact.capture = capture
        if args.surface_diagnostics:
            from inspire_sweep_baseline.mdp.reward_functions.sweep_contact_preference import sweep_region
            from inspire_sweep_baseline.mdp.common.revision_goal import object_geometry_in_shelf
            from inspire_sweep_baseline.mdp.common.contact_sweep_clutter import all_object_tilt_angles
            surface_term = raw.reward_manager.get_term_cfg('sweep_progress').func
            original_surface_loads = surface_term._surface_loads

            def surface_loads(source):
                palm_load, dorsal_load = original_surface_loads(source)
                for name, value in (('palm_load', palm_load), ('dorsal_load', dorsal_load)):
                    all_rows.setdefault(name, []).append(value.detach().cpu().numpy().copy())
                return palm_load, dorsal_load

            surface_term._surface_loads = surface_loads
        timeline_probe = None
        if args.reward_timeline:
            from reward_timeline_probe import RewardTimelineProbe
            timeline_probe = RewardTimelineProbe(raw, all_rows, args.output_dir, report)
        original_compute = raw.termination_manager.compute

        def compute():
            nonlocal initial_target, initial_goal, previous_distance
            done = original_compute()
            if len(substeps) != cfg.decimation:
                raise RuntimeError(f'Incomplete physics samples: {len(substeps)}')
            normal, friction, weighted, load, pair_load = [torch.stack([s[i] for s in substeps]).mean(0) for i in range(5)]
            substeps.clear()
            target = point(centroid(raw))
            goal = point(raw.command_manager.get_command('target_goal_pos')[:, :3])
            eef = raw.scene['ee_frame'].data
            palm = quat_apply(eef.target_quat_w[:, 0], target.new_tensor([0., 0., 1.]).expand(n, -1))
            distance = (target[:, :2] - goal[:, :2]).norm(dim=-1)
            recognized = ((raw._stabilize_state.contact_history.sum(-1) >= 2)
                          & (eef.target_pos_w[:, 0, 2] < origins[:, 2] + 1.05 + cfg.sweep_contact_height_limit)
                          & contact.all_valid)
            # Initial positions come from the exact pre-step observation, updated below.
            rows = dict(episode=episode, step=raw.episode_length_buf, target=target,
                        goal=goal, initial_target=initial_target, initial_goal=initial_goal,
                        eef=point(eef.target_pos_w[:, 0]), palm=vector(palm),
                        normal=vector(normal), friction=vector(friction), force=vector(contact.force_sum / contact.samples),
                        contact_point=point(weighted / load[:, None].clamp_min(1e-12)),
                        load=load, pair_load=pair_load, valid=contact.all_valid,
                        contact=contact.contact, recognized=recognized,
                        positive=raw._stabilize_state.positive, retreat=raw._stabilize_state.retreat,
                        distance=distance, delta_distance=previous_distance - distance,
                        joint_pos=raw.scene['robot'].data.joint_pos[:, hand_ids],
                        hand_points=quat_apply_inverse(rotation[:, None].expand(-1, len(body_ids), -1),
                            raw.scene['robot'].data.body_link_pos_w[:, body_ids] - origins[:, None]),
                        velocity=vector(raw.scene['object_collection'].data.object_com_lin_vel_w[:, 0]),
                        action=raw.action_manager.action, done=done,
                        success=raw.termination_manager.get_term('success'))
            if hasattr(raw._stabilize_state, 'contact_count'):
                rows['contact_count'] = raw._stabilize_state.contact_count
            else:
                rows['contact_recognized'] = raw._stabilize_state.contact_recognized
            if args.surface_diagnostics:
                shelf = raw.scene['shelf'].data
                _, bottom = object_geometry_in_shelf(raw)
                rows.update(region=sweep_region(quat_apply_inverse(shelf.root_quat_w, raw.sweep_dir)),
                            sweep_direction=vector(raw.sweep_dir), object_bottom=bottom,
                            object_tilt=all_object_tilt_angles(raw), active_objects=raw._active_object_mask,
                            object_positions=raw.scene['object_collection'].data.object_pos_w - origins[:, None],
                            shelf_velocity=shelf.root_vel_w.norm(dim=-1),
                            shelf_displacement=(shelf.root_pos_w - origins).norm(dim=-1),
                            wrist_clearance=raw.scene['wrist_frame'].data.target_pos_w[:, 0, 2] - shelf.root_pos_w[:, 2] - 1.05,
                            arm_peak_velocity=raw.scene['robot'].data.joint_vel[:, arm_ids].abs().amax(-1))
            for name in ('object_fall_or_tilt', 'shelf_failure', 'arm_joint_velocity', 'timeout'):
                rows[name] = raw.termination_manager.get_term(name)
            if timeline_probe is not None:
                rows.update(timeline_probe.state_inputs())
            for name, value in rows.items():
                all_rows.setdefault(name, []).append(value.detach().cpu().numpy().copy())
            previous_distance = distance.clone()
            episode.add_(done.long())
            return done

        raw.termination_manager.compute = compute
        obs = env.get_observations()
        if timeline_probe is not None:
            timeline_probe.save_initial(obs)
        start = time.perf_counter()
        for step in range(args.steps):
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
                policy.reset(done)
                # Refresh only reset rows; never include auto-reset positions in the trace.
                ids = done.nonzero(as_tuple=False).flatten()
                if len(ids):
                    initial_target[ids] = point(centroid(raw))[ids]
                    initial_goal[ids] = point(raw.command_manager.get_command('target_goal_pos')[:, :3])[ids]
                    previous_distance[ids] = (initial_target[ids, :2] - initial_goal[ids, :2]).norm(dim=-1)
            if (step + 1) % 100 == 0:
                completed = sum(int(x.sum()) for x in all_rows['done'])
                successes = sum(int(x.sum()) for x in all_rows['success'])
                print(f'[Analysis] {step+1}/{args.steps}; complete={completed}; success={successes}; elapsed={time.perf_counter()-start:.1f}s', flush=True)
        if any(len(v) != args.steps for v in all_rows.values()):
            raise RuntimeError('Analysis columns have inconsistent policy-step counts')
        np.savez_compressed(args.output_dir / 'trace.npz', **{k: np.stack(v) for k, v in all_rows.items()})
        report.update(status='complete', dt=raw.step_dt, steps=args.steps,
                      checkpoint_iteration=runner.current_learning_iteration,
                      elapsed_s=time.perf_counter()-start, filter_body_names=list(HAND_CONTACT_BODY_NAMES),
                      hand_drive_names=list(HAND_DRIVE_JOINT_NAMES),
                      hand_body_names=list(HAND_LINK_BODY_NAMES),
                      max_contact_points_per_filter=max_count, normal_matrix_max_abs_error_n=force_error,
                      completed_episodes=int(np.stack(all_rows['done']).sum()),
                      successes=int(np.stack(all_rows['success']).sum()),
                      incomplete_episodes_excluded_from_outcome_statistics=True,
                      force_definition='force ON target FROM hand = normal + friction; averaged over physics substeps',
                      contact_point_definition='normal-load weighted contact-point centroid over physics substeps',
                      palm_definition='EEF +Z; orientation proxy, not exact mesh-side contact classification')
    except BaseException as exc:
        import traceback
        report.update(status='failed', error=repr(exc))
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], file=sys.stderr, flush=True)
        raise
    finally:
        (args.output_dir / 'metadata.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
