#!/usr/bin/env python3
"""Check Push contracts and a scripted physical fixture, independently of learned policy quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from itertools import product
import math
from pathlib import Path
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
PUSH_TASK_ID = "Isaac-Hand-Manipulation-Push-v0"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(PUSH_TASK_ID,), default=PUSH_TASK_ID)
parser.add_argument("--num_envs", "--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=50, help="Normal rollout steps with zero arm delta and the initial open hand.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--reset-samples", type=int, default=4)
parser.add_argument("--check-timeout", action="store_true")
parser.add_argument("--check-hand-range", action="store_true", help="Exercise both hand action endpoints and verify all twelve physical joint limits.")
parser.add_argument("--debug-vis", action="store_true")
parser.add_argument("--disable-markers", action="store_true")
parser.add_argument("--skip-physical-fixture", action="store_true", help="Run manager/reset checks without the separate scripted physical fixture.")
parser.add_argument("--physical-only", action="store_true", help="Run the scripted fixture after one certified reset, skipping boundary/rollout/timeout checks.")
parser.add_argument("--prescribed-robot-fixture", action="store_true", help="Use prescribed hand geometry for the separate physical fixture instead of OSC; does not validate actuator pushing strength.")
parser.add_argument("--physical-steps", type=int, default=350, help="Maximum scripted side-push steps, below the episode timeout.")
parser.add_argument("--profile-only", action="store_true", help="Compare synchronized reset and rollout costs without the expensive contract/physical fixtures.")
parser.add_argument("--profile-steps", type=int, default=36, help="Uninstrumented policy steps in each profiling window.")
parser.add_argument("--profile-reset-backend", choices=("cached", "numpy", "torch", "physx", "both", "all"), default="both", help="Compare original PhysX, cached CPU NumPy, device Torch, or automatically selected reset backends.")
parser.add_argument("--check-reset-kinematics", action="store_true", help="Compare cached FK/Jacobians against actual PhysX finite differences at each checked reset.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = launcher.app


def main() -> None:
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import axis_angle_from_quat, compute_pose_error, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from pxr import UsdPhysics

    import hand_manipulation_test  # noqa: F401 -- register the tasks
    from hand_manipulation_test.action_math import inspire_synergy_to_joint_positions
    from hand_manipulation_test.assets.robot import HAND_CLOSED_TARGETS, HAND_JOINT_NAMES, HAND_OPEN_TARGETS, ROBOT_CONTACT_BODY_NAMES
    from hand_manipulation_test.constants import PUSH_OBSERVATION_LAYOUT, PUSH_OBSERVATION_SLICES
    from hand_manipulation_test.geometry import control_point_pose_w, palm_tactile_bits, wrist_wrench_c
    from hand_manipulation_test.sensors import PALM_CHANNEL_NAMES

    if args.num_envs <= 0 or args.reset_samples < 0:
        raise ValueError("--num_envs must be positive and --reset-samples nonnegative")
    if not 50 <= args.steps < 500 or not 1 <= args.physical_steps <= 450:
        raise ValueError("--steps must be in [50,499], --physical-steps in [1,450]")
    if args.physical_only and args.skip_physical_fixture:
        raise ValueError("--physical-only and --skip-physical-fixture cannot be combined")
    if args.prescribed_robot_fixture and args.skip_physical_fixture:
        raise ValueError("--prescribed-robot-fixture requires the physical fixture")
    if args.profile_steps <= 0:
        raise ValueError("--profile-steps must be positive")
    cfg = load_cfg_from_registry(PUSH_TASK_ID, "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.commands.target_position.debug_vis = args.debug_vis and not args.disable_markers
    cfg.scene.ee_frame.debug_vis = args.debug_vis and not args.disable_markers
    env = gym.make(PUSH_TASK_ID, cfg=cfg).unwrapped
    try:
        robot, cube = env.scene["robot"], env.scene["target_object"]
        command = env.command_manager.get_term("target_position")
        safe = env.event_manager.get_term_cfg("safe_hand").func
        all_ids = torch.arange(env.num_envs, device=env.device)
        reset_seconds = []
        fixture_cube_position = (
            (cfg.task.object_xy_range_low[0] + cfg.task.object_xy_range_high[0]) / 2,
            (cfg.task.object_xy_range_low[1] + cfg.task.object_xy_range_high[1]) / 2,
            cfg.task.cube_center_height_m,
        )

        def finite(name, value):
            assert bool(torch.isfinite(value).all()), f"Non-finite {name}"

        def part(observation, name):
            return observation["policy"][:, PUSH_OBSERVATION_SLICES[name]]

        def hold_action():
            action = torch.zeros((env.num_envs, 8), device=env.device)
            hand = env.action_manager.get_term("hand_action")
            action[:, 6:] = hand.synergy_to_action(hand.actual_synergy)
            return action

        def counters():
            return {name: getattr(safe, name).clone() for name in ("attempts", "accepted_seed_index", "ik_iterations", "obb_checks")}

        def same_counters(before, ids=None):
            ids = all_ids if ids is None else ids
            for name, after in counters().items():
                torch.testing.assert_close(after[ids], before[name][ids], rtol=0.0, atol=0.0)

        def snapshot_state():
            state = command.state()
            return {field.name: getattr(state, field.name).clone() for field in fields(state)}

        def check_command():
            local = command.target_pos_w - env.scene.env_origins
            low = local.new_tensor(cfg.task.object_xy_range_low)
            high = local.new_tensor(cfg.task.object_xy_range_high)
            assert bool(((local[:, :2] >= low - 2e-6) & (local[:, :2] <= high + 2e-6)).all())
            torch.testing.assert_close(local[:, 2], torch.full_like(local[:, 2], cfg.task.cube_center_height_m), rtol=0.0, atol=2e-6)
            relative_angle = torch.remainder(command.angle_rad + math.pi / 2, math.pi) - math.pi / 2
            assert bool((relative_angle.abs() <= cfg.task.command_angle_jitter_rad + 2e-6).all())
            low_distance, high_distance = cfg.task.command_distance_range
            assert bool(((command.distance_m >= low_distance - 1e-6) & (command.distance_m <= high_distance + 1e-6)).all())
            torch.testing.assert_close(command.goal_pos_w, command.target_pos_w + command.distance_m[:, None] * command.direction_w, rtol=0.0, atol=2e-6)
            # Independently certify the entire segment in the convex Table
            # rectangle, including a radius covering every Cube orientation.
            centre = local.new_tensor(cfg.task.table_center[:2])
            radius = math.sqrt(3.0) * cfg.task.cube_size / 2 + cfg.task.table_path_margin_m
            extent = local.new_tensor(cfg.task.table_size[:2]) / 2 - radius
            for position in (command.target_pos_w, command.goal_pos_w):
                relative = position[:, :2] - env.scene.env_origins[:, :2] - centre
                assert bool((relative.abs() <= extent + 2e-6).all()), relative.tolist()
            base_quaternion = env.scene["ee_frame"].data.source_quat_w
            expected_direction = quat_apply(base_quaternion, torch.stack((command.angle_rad.sin(), -command.angle_rad.cos(), torch.zeros_like(command.angle_rad)), -1))
            torch.testing.assert_close(command.direction_w, expected_direction, rtol=0.0, atol=2e-6)

        def check_policy(observation, reset_ids=None):
            assert observation["policy"].shape == (env.num_envs, 61), observation["policy"].shape
            finite("61D policy", observation["policy"])
            actual_names = tuple(env.observation_manager.active_terms["policy"])
            assert actual_names == tuple(name for name, _ in PUSH_OBSERVATION_LAYOUT), actual_names
            torch.testing.assert_close(part(observation, "surface_header_and_tactile")[:, 0], torch.zeros(env.num_envs, device=env.device), rtol=0.0, atol=0.0)
            if reset_ids is not None:
                for name in ("surface_header_and_tactile", "wrist_wrench_c", "last_action"):
                    value = part(observation, name)[reset_ids]
                    torch.testing.assert_close(value, torch.zeros_like(value), rtol=0.0, atol=0.0)
            else:
                torch.testing.assert_close(part(observation, "surface_header_and_tactile")[:, 1:], palm_tactile_bits(env), rtol=0.0, atol=0.0)
                measured = wrist_wrench_c(env).measured_c
                expected = torch.cat(((measured[:, :3] / cfg.task.wrench_force_observation_scale_n).clamp(-1, 1),
                                      (measured[:, 3:] / cfg.task.wrench_moment_observation_scale_nm).clamp(-1, 1)), -1)
                torch.testing.assert_close(part(observation, "wrist_wrench_c"), expected, rtol=0.0, atol=1e-6)
            frame = env.scene["ee_frame"].data
            torch.testing.assert_close(part(observation, "initial_target_relative_position"), quat_apply_inverse(frame.source_quat_w, command.target_pos_w - frame.source_pos_w), rtol=1e-5, atol=2e-6)
            torch.testing.assert_close(part(observation, "current_cube_base_position"), quat_apply_inverse(frame.source_quat_w, cube.data.root_pos_w - frame.source_pos_w), rtol=1e-5, atol=2e-6)
            expected_command = torch.stack((command.angle_rad.sin(), command.angle_rad.cos(), command.distance_m), -1)
            torch.testing.assert_close(part(observation, "push_command"), expected_command, rtol=0.0, atol=1e-6)
            position, quaternion = control_point_pose_w(env)
            torch.testing.assert_close(frame.target_pos_w[:, 0], position, rtol=1e-5, atol=2e-6)
            cosine = (frame.target_quat_w[:, 0] * quaternion).sum(-1).abs()
            torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=0.0, atol=2e-6)

        def check_initial(observation, ids):
            check_policy(observation, reset_ids=ids)
            check_command()
            assert bool(safe.check_final(ids).all()), "Initial palm pose or all-robot 4mm OBB clearance failed"
            cached_position, cached_quaternion = safe.cached_control_pose_w(ids)
            actual_position, actual_quaternion = control_point_pose_w(env)
            torch.testing.assert_close(cached_position, actual_position[ids], rtol=0., atol=5.e-6)
            torch.testing.assert_close((cached_quaternion * actual_quaternion[ids]).sum(-1).abs(),
                                      torch.ones(len(ids), device=env.device), rtol=0., atol=2.e-6)
            if args.check_reset_kinematics:
                cached_jacobian = safe.cached_numerical_control_jacobian(ids)
                physical_jacobian = safe._numerical_control_jacobian(ids)
                torch.testing.assert_close(cached_jacobian, physical_jacobian, rtol=2.e-3, atol=1.e-3)
            assert abs(cfg.task.reset_clearance_margin_m - 0.004) < 1e-8
            assert safe.collision_bounds.centers.shape[0] == len(ROBOT_CONTACT_BODY_NAMES)
            assert not bool(safe.last_cube_overlap[ids].any())
            assert not bool(safe.last_table_overlap[ids].any())
            seed_count = len(cfg.task.reset_joint_seed_offsets)
            assert seed_count == 3
            assert bool(((safe.attempts[ids] >= 1) & (safe.attempts[ids] <= seed_count)).all())
            assert bool(((safe.accepted_seed_index[ids] >= 0) & (safe.accepted_seed_index[ids] < seed_count)).all())
            assert bool((safe.ik_iterations[ids] <= safe.attempts[ids] * cfg.task.reset_max_iterations).all())
            assert bool((safe.obb_checks[ids] <= safe.attempts[ids]).all())
            palm, hand_quaternion = safe.palm_reference_pose_w(ids)
            normal = quat_apply(hand_quaternion, palm.new_tensor((0., 1., 0.)).expand(len(ids), -1))
            finger = quat_apply(hand_quaternion, palm.new_tensor((0., 0., 1.)).expand(len(ids), -1))
            tolerance = cfg.task.reset_orientation_tolerance_rad
            assert bool(((normal * command.direction_w[ids]).sum(-1) >= math.cos(tolerance) - 1e-6).all())
            assert bool((normal[:, 2].abs() <= math.sin(tolerance) + 1e-6).all())
            preferred = palm.new_tensor((-1., 0., 0.)).expand(len(ids), -1)
            expected_finger = preferred - (preferred * command.direction_w[ids]).sum(-1, keepdim=True) * command.direction_w[ids]
            expected_finger = torch.nn.functional.normalize(expected_finger, dim=-1)
            assert bool(((finger * expected_finger).sum(-1) >= math.cos(tolerance) - 1e-6).all()), "Fingers do not follow projected outward -X"
            outward = env.scene["table"].data.root_pos_w[ids] - robot.data.root_link_pos_w[ids]
            outward[:, 2] = 0.
            assert bool(((finger[:, :2] * outward[:, :2]).sum(-1) > 0.).all()), "Fingers point toward the actual robot base"
            table_local = env.scene["table"].data.root_pos_w[ids] - env.scene.env_origins[ids]
            torch.testing.assert_close(table_local, table_local.new_tensor(cfg.task.table_center).expand(len(ids), -1), rtol=0., atol=5e-6)
            delta = palm - command.target_pos_w[ids]
            backoff = -(delta * command.direction_w[ids]).sum(-1)
            allowance = cfg.task.reset_position_tolerance_m
            assert bool(((backoff - cfg.task.reset_position_offset_task[0]).abs() <= cfg.task.reset_position_jitter_task[0] + allowance + 1e-5).all())
            assert bool(((delta[:, 2] - cfg.task.reset_position_offset_task[2]).abs() <= cfg.task.reset_position_jitter_task[2] + allowance + 1e-5).all())
            torch.testing.assert_close(robot.data.joint_vel[ids], torch.zeros_like(robot.data.joint_vel[ids]), rtol=0.0, atol=1e-6)
            torch.testing.assert_close(cube.data.root_pos_w[ids], command.target_pos_w[ids], rtol=0.0, atol=2e-6)
            torch.testing.assert_close(cube.data.root_vel_w[ids], torch.zeros_like(cube.data.root_vel_w[ids]), rtol=0.0, atol=1e-6)
            synergy = part(observation, "hand_state")[ids]
            low, high = cfg.task.initial_hand_open_range
            assert bool(((synergy >= low - 1e-6) & (synergy <= high + 1e-6)).all())
            hand_ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
            expected = inspire_synergy_to_joint_positions(synergy, synergy.new_tensor([HAND_OPEN_TARGETS[name] for name in HAND_JOINT_NAMES]), synergy.new_tensor([HAND_CLOSED_TARGETS[name] for name in HAND_JOINT_NAMES]))
            torch.testing.assert_close(robot.data.joint_pos[ids][:, hand_ids], expected, rtol=0.0, atol=1e-6)
            state = command.state()
            for name in ("contact_seen", "push_seen", "success", "failure", "valid_push_contact"):
                assert not bool(getattr(state, name)[ids].any()), name
            torch.testing.assert_close(state.settled_time_s[ids], torch.zeros_like(state.settled_time_s[ids]), rtol=0.0, atol=0.0)

        def reset(*, seed=None):
            before_counter = env._sim_step_counter
            started = time.perf_counter()
            observation, _ = env.reset(seed=seed)
            reset_seconds.append(time.perf_counter() - started)
            assert env._sim_step_counter == before_counter, "Reset IK must not step simulation"
            check_initial(observation, all_ids)
            return observation

        if args.profile_only:
            def synchronize():
                if torch.device(env.device).type == "cuda":
                    torch.cuda.synchronize(env.device)

            def timed(function):
                synchronize()
                started = time.perf_counter()
                result = function()
                synchronize()
                return time.perf_counter() - started, result

            def io_delta(before):
                return {name: count - before[name] for name, count in safe.reset_io_counters.items()}

            backends = (("physx", "numpy", "torch") if args.profile_reset_backend == "all" else
                        ("physx", "cached") if args.profile_reset_backend == "both" else (args.profile_reset_backend,))
            row_counts = sorted(set((1, max(1, env.num_envs // 16), env.num_envs)))
            results = []
            original_backend = safe.use_cached_kinematics
            original_selection = safe.reset_backend
            try:
                for backend in backends:
                    safe.use_cached_kinematics = backend != "physx"
                    safe.reset_backend = "auto" if backend == "cached" else backend
                    # Exclude warmup from measurements and use the same seeds
                    # and pending row batches for the two implementations.
                    env.reset(seed=args.seed)
                    action = hold_action()
                    for _ in range(3):
                        env.step(action)
                    backend_results = {"backend": backend, "resets": []}
                    for count in row_counts:
                        ids = all_ids[:count]
                        for sample in range(max(1, args.reset_samples)):
                            before_io = dict(safe.reset_io_counters)
                            elapsed, _ = timed(lambda: env.reset(seed=args.seed + sample + 1, env_ids=ids))
                            instrumentation = counters()
                            backend_results["resets"].append({
                                "rows": count, "sample": sample, "seconds": elapsed,
                                "io_calls": io_delta(before_io),
                                "attempts_sum": int(instrumentation["attempts"][ids].sum()),
                                "ik_iterations_sum": int(instrumentation["ik_iterations"][ids].sum()),
                                "obb_checks_sum": int(instrumentation["obb_checks"][ids].sum()),
                            })
                    before_io = dict(safe.reset_io_counters)
                    action = hold_action()
                    def rollout():
                        for _ in range(args.profile_steps):
                            env.step(action)
                    elapsed, _ = timed(rollout)
                    backend_results["rollout"] = {
                        "steps": args.profile_steps, "seconds": elapsed,
                        "seconds_per_policy_step": elapsed / args.profile_steps,
                        "environment_steps_per_second": args.profile_steps * env.num_envs / elapsed,
                        "reset_io_calls": io_delta(before_io),
                    }
                    results.append(backend_results)
            finally:
                safe.use_cached_kinematics = original_backend
                safe.reset_backend = original_selection
            print("PROFILE_PUSH " + json.dumps({
                "num_envs": env.num_envs, "physics_dt": env.physics_dt,
                "policy_dt": env.step_dt, "markers": cfg.commands.target_position.debug_vis,
                "results": results,
            }), flush=True)
            return

        def step(action):
            observation, reward, terminated, truncated, extras = env.step(action)
            if bool((terminated | truncated).any()):
                raise AssertionError(f"Unexpected episode end: terminated={terminated.tolist()}, truncated={truncated.tolist()}")
            check_policy(observation)
            finite("reward", reward)
            finite("robot positions", robot.data.joint_pos)
            finite("robot velocities", robot.data.joint_vel)
            first_state = command.state()
            second_state = command.state()
            for field in fields(first_state):
                torch.testing.assert_close(getattr(first_state, field.name), getattr(second_state, field.name), rtol=0.0, atol=0.0)
            for index in range(env.num_envs):
                terms = {name: values[0] for name, values in env.reward_manager.get_active_iterable_terms(index)}
                torch.testing.assert_close(reward[index], reward.new_tensor(sum(terms.values()) * env.step_dt), rtol=1e-5, atol=1e-6)
            return observation, reward, first_state

        if args.physical_only:
            fixture_angles = torch.tensor([0. if index % 2 == 0 else math.pi for index in range(env.num_envs)], device=env.device)
            command.set_reset_specs(all_ids, fixture_cube_position, fixture_angles, cfg.task.command_distance_range[0])
        observation = reset(seed=args.seed)
        assert env.action_manager.total_action_dim == 8
        assert cfg.actions.arm_action.motion_stiffness == (200.0,) * 6
        assert cfg.actions.hand_action.synergy_range == (0.8, 1.0)
        assert cfg.actions.hand_action.enforce_synergy_joint_limits
        assert env.max_episode_length == 500 and abs(env.step_dt - .02) < 1e-9
        assert set(env.reward_manager.active_terms) == {"approach", "progress", "backslide", "first_contact", "contact", "alignment", "action_rate", "success", "failure"}
        assert set(env.termination_manager.active_terms) == {"failure", "success", "time_out"}
        assert set(env.scene.sensors) == {"palm_tactile", "ee_frame", "table_contacts", "cube_palm_contacts", "cube_table_contacts"}
        assert set(env.scene["palm_tactile"].body_names) == set(PALM_CHANNEL_NAMES)
        for name in ("table_contacts", "cube_palm_contacts", "cube_table_contacts"):
            assert getattr(cfg.scene, name).history_length == 2
        if args.num_envs > 1:
            assert bool((env.scene.env_origins[0] != env.scene.env_origins[1]).any())

        # One-shot fixtures are checked before IK; rejected commands are atomic.
        before_cube = cube.data.root_pose_w.clone()
        before_command = command.command.clone()
        before_instrumentation = counters()
        try:
            command.set_reset_specs(all_ids, fixture_cube_position, math.pi / 2, sum(cfg.task.command_distance_range) / 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-band Push angle was accepted")
        torch.testing.assert_close(cube.data.root_pose_w, before_cube, rtol=0.0, atol=0.0)
        torch.testing.assert_close(command.command, before_command, rtol=0.0, atol=0.0)
        same_counters(before_instrumentation)
        angles = tuple(math.radians(value) for value in (-10., 0., 10., 170., 180., 190.))
        corners = tuple(product((cfg.task.object_xy_range_low[0], cfg.task.object_xy_range_high[0]), (cfg.task.object_xy_range_low[1], cfg.task.object_xy_range_high[1])))
        specs = [(x, y, angle, cfg.task.command_distance_range[1]) for (x, y), angle in product(corners, angles)]
        specs += [(*fixture_cube_position[:2], angle, cfg.task.command_distance_range[0]) for angle in (0., math.pi)]
        if args.physical_only:
            specs = []
        for offset in range(0, len(specs), env.num_envs):
            batch = [specs[(offset + index) % len(specs)] for index in range(env.num_envs)]
            positions = torch.tensor([(x, y, cfg.task.cube_center_height_m) for x, y, _, _ in batch], device=env.device)
            requested_angle = torch.tensor([angle for _, _, angle, _ in batch], device=env.device)
            requested_distance = torch.tensor([distance for _, _, _, distance in batch], device=env.device)
            command.set_reset_specs(all_ids, positions, requested_angle, requested_distance)
            observation = reset()
            torch.testing.assert_close(command.target_pos_w - env.scene.env_origins, positions, rtol=0.0, atol=2e-6)
            torch.testing.assert_close(command.angle_rad, requested_angle, rtol=0.0, atol=1e-6)
            torch.testing.assert_close(command.distance_m, requested_distance, rtol=0.0, atol=1e-6)
        for _ in range(0 if args.physical_only else args.reset_samples):
            observation = reset()

        if cfg.commands.target_position.debug_vis:
            command._debug_vis_callback(None)
            assert command._target_marker.count == env.num_envs
            assert command._goal_marker.count == env.num_envs
            assert command._direction_marker.count == 2 * env.num_envs
        if cfg.scene.ee_frame.debug_vis:
            env.scene["ee_frame"]._debug_vis_callback(None)
            assert env.scene["ee_frame"].frame_visualizer.count == 3 * env.num_envs
        for prim in env.sim.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/Visuals/HandManipulation"):
                assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI), path
            assert "dorsal" not in prim.GetName().lower(), path

        if not args.physical_only:
            observation = reset()
        initial_target, fixed_goal, fixed_command = command.target_pos_w.clone(), command.goal_pos_w.clone(), command.command.clone()
        initial_observation = part(observation, "initial_target_relative_position").clone()
        instrumentation = counters()
        action = hold_action()
        peak_tracking_position = torch.zeros(env.num_envs, device=env.device)
        peak_tracking_angle = torch.zeros_like(peak_tracking_position)
        force_peak = torch.zeros_like(peak_tracking_position)
        started = time.perf_counter()
        for _ in range(0 if args.physical_only else args.steps):
            observation, _, _ = step(action)
            torch.testing.assert_close(command.target_pos_w, initial_target, rtol=0.0, atol=0.0)
            torch.testing.assert_close(command.goal_pos_w, fixed_goal, rtol=0.0, atol=0.0)
            torch.testing.assert_close(command.command, fixed_command, rtol=0.0, atol=0.0)
            torch.testing.assert_close(part(observation, "initial_target_relative_position"), initial_observation, rtol=0.0, atol=2e-6)
            position, quaternion = control_point_pose_w(env)
            current_position_b = quat_apply_inverse(robot.data.root_link_quat_w, position - robot.data.root_link_pos_w)
            current_quaternion_b = quat_mul(quat_conjugate(robot.data.root_link_quat_w), quaternion)
            desired = env.action_manager.get_term("arm_action").desired_c_pose_b
            peak_tracking_position = torch.maximum(peak_tracking_position, (current_position_b - desired[:, :3]).norm(dim=-1))
            cosine = (current_quaternion_b * desired[:, 3:]).sum(-1).abs().clamp(0., 1.)
            peak_tracking_angle = torch.maximum(peak_tracking_angle, 2. * torch.acos(cosine))
            sample = wrist_wrench_c(env)
            torch.testing.assert_close(sample.raw_f, sample.measured_f, rtol=0.0, atol=0.0)
            force_peak = torch.maximum(force_peak, sample.raw_f[:, :3].norm(dim=-1))
        step_seconds = (time.perf_counter() - started) / args.steps
        same_counters(instrumentation)
        if not args.physical_only:
            assert bool((peak_tracking_position < .02).all())
            assert bool((peak_tracking_angle < .05).all())
            assert bool((force_peak > .01).all()), "F/T discarded hand gravity"

        # Actual Cube motion must affect only the current-Cube position slice.
        if not args.physical_only:
            pose = cube.data.root_pose_w.clone()
            pose[:, :3] += .02 * command.direction_w
            cube.write_root_pose_to_sim(pose)
            cube.write_root_velocity_to_sim(torch.zeros_like(cube.data.root_vel_w))
            observation, _, _ = step(hold_action())
            torch.testing.assert_close(command.target_pos_w, initial_target, rtol=0.0, atol=0.0)
            torch.testing.assert_close(command.goal_pos_w, fixed_goal, rtol=0.0, atol=0.0)
            torch.testing.assert_close(part(observation, "initial_target_relative_position"), initial_observation, rtol=0.0, atol=2e-6)
            assert bool(((cube.data.root_pos_w - initial_target) * command.direction_w).sum(-1).gt(.015).all())

        if env.num_envs > 1 and not args.physical_only:
            untouched_ids = all_ids[1:]
            before = snapshot_state()
            untouched_goal = command.goal_pos_w[1:].clone()
            untouched_initial = command.target_pos_w[1:].clone()
            untouched_cube = cube.data.root_state_w[1:].clone()
            untouched_robot_position = robot.data.joint_pos[1:].clone()
            untouched_robot_velocity = robot.data.joint_vel[1:].clone()
            untouched_action = part(observation, "last_action")[1:].clone()
            instrumentation = counters()
            observation, _ = env.reset(env_ids=all_ids[:1])
            check_initial(observation, all_ids[:1])
            after = snapshot_state()
            for name in before:
                torch.testing.assert_close(after[name][1:], before[name][1:], rtol=0.0, atol=0.0)
            torch.testing.assert_close(command.goal_pos_w[1:], untouched_goal, rtol=0.0, atol=0.0)
            torch.testing.assert_close(command.target_pos_w[1:], untouched_initial, rtol=0.0, atol=0.0)
            torch.testing.assert_close(cube.data.root_state_w[1:], untouched_cube, rtol=0.0, atol=0.0)
            torch.testing.assert_close(robot.data.joint_pos[1:], untouched_robot_position, rtol=0.0, atol=0.0)
            torch.testing.assert_close(robot.data.joint_vel[1:], untouched_robot_velocity, rtol=0.0, atol=0.0)
            torch.testing.assert_close(part(observation, "last_action")[1:], untouched_action, rtol=0.0, atol=0.0)
            same_counters(instrumentation, untouched_ids)
            observation, _, _ = step(hold_action())
            rate = {name: values[0] for name, values in env.reward_manager.get_active_iterable_terms(0)}["action_rate"]
            assert abs(rate) < 1e-9

        if args.check_timeout and not args.physical_only:
            observation = reset()
            action = hold_action()
            initial_target, fixed_goal, fixed_command = command.target_pos_w.clone(), command.goal_pos_w.clone(), command.command.clone()
            for index in range(500):
                observation, reward, terminated, truncated, _ = env.step(action)
                finite("timeout observation", observation["policy"])
                finite("timeout reward", reward)
                assert not bool(terminated.any()), f"Unexpected timeout fixture failure at {index}"
                if index < 499:
                    assert not bool(truncated.any()), index
                    torch.testing.assert_close(command.target_pos_w, initial_target, rtol=0.0, atol=0.0)
                    torch.testing.assert_close(command.goal_pos_w, fixed_goal, rtol=0.0, atol=0.0)
                    torch.testing.assert_close(command.command, fixed_command, rtol=0.0, atol=0.0)
                else:
                    assert bool(truncated.all()), "500-step timeout did not reset every environment"
                    check_initial(observation, all_ids)

        if args.check_hand_range and not args.physical_only:
            hand = env.action_manager.get_term("hand_action")
            hand_ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
            opened = robot.data.joint_pos.new_tensor([HAND_OPEN_TARGETS[name] for name in HAND_JOINT_NAMES])
            closed = robot.data.joint_pos.new_tensor([HAND_CLOSED_TARGETS[name] for name in HAND_JOINT_NAMES])
            endpoints = inspire_synergy_to_joint_positions(
                opened.new_tensor(((0.8, 0.8), (1.0, 1.0))), opened, closed
            )
            limits = torch.stack((endpoints.amin(0), endpoints.amax(0)), dim=-1)
            actual_limits = robot.root_physx_view.get_dof_limits()[:, hand_ids].to(env.device)
            torch.testing.assert_close(actual_limits, limits.expand(env.num_envs, -1, -1), rtol=0, atol=1e-6)
            minimum = torch.full((2,), float("inf"), device=env.device)
            maximum = torch.full((2,), -float("inf"), device=env.device)
            for endpoint in (-1.0, 1.0):
                reset()
                action = torch.zeros((env.num_envs, 8), device=env.device)
                action[:, 6:] = endpoint
                for _ in range(50):
                    step(action)
                    q = robot.data.joint_pos[:, hand_ids]
                    assert bool((q >= limits[:, 0] - 0.002).all()), "Hand crossed the physical open limit"
                    assert bool((q <= limits[:, 1] + 0.002).all()), "Hand crossed the physical closed limit"
                    assert bool(((hand.synergy_target >= 0.8 - 1e-6) & (hand.synergy_target <= 1.0 + 1e-6)).all())
                    minimum = torch.minimum(minimum, hand.actual_synergy.amin(0))
                    maximum = torch.maximum(maximum, hand.actual_synergy.amax(0))
                torch.testing.assert_close(hand.synergy_target, torch.full_like(hand.synergy_target, .8 if endpoint < 0 else 1.0), rtol=0, atol=1e-5)
            print("[PASS] Push physical hand range", {"actual_min": minimum.tolist(), "actual_max": maximum.tolist()}, flush=True)

        fixture_result = None
        if not args.skip_physical_fixture:
            if not args.physical_only:
                fixture_angles = torch.tensor([0. if index % 2 == 0 else math.pi for index in range(env.num_envs)], device=env.device)
                command.set_reset_specs(all_ids, fixture_cube_position, fixture_angles, cfg.task.command_distance_range[0])
                observation = reset()
            instrumentation = counters()
            action = hold_action()
            direction = command.direction_w.clone()
            physical_initial = cube.data.root_pos_w.clone()
            ever_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            ever_push = torch.zeros_like(ever_contact)
            peak_forward = torch.zeros(env.num_envs, device=env.device)
            translation_scale = action.new_tensor(cfg.actions.arm_action.translation_scale)
            rotation_scale = action.new_tensor(cfg.actions.arm_action.rotation_scale)
            _, initial_quaternion = control_point_pose_w(env)
            initial_quaternion = initial_quaternion.clone()
            initial_control_position = control_point_pose_w(env)[0].clone()
            initial_palm_height = safe.palm_reference_pose_w()[0][:, 2].clone()
            prescribed_distance = torch.zeros(env.num_envs, device=env.device)
            fixture_ik_iterations = 0
            fixture_obb_checks = 0
            original_state = command.state
            recorded = {}

            def recording_state():
                # Retain the final live transition before ManagerBasedRLEnv
                # auto-resets terminal rows. Only this fixture is instrumented.
                value = original_state()
                live = env.episode_length_buf > 0
                samples = {name: getattr(value, name) for name in (
                    "grounded", "valid_push_contact", "contact_seen", "push_seen", "success", "failure",
                    "footprint_failure", "fall_failure", "table_failure", "palm_alignment_cos", "force_alignment_cos",
                )}
                samples["cube_pose_w"] = cube.data.root_pose_w
                samples["cube_velocity_w"] = cube.data.root_vel_w
                samples["cube_table_force_w"] = env.scene["cube_table_contacts"].data.force_matrix_w[:, 0, 0]
                samples["cube_palm_pad_force_n"] = env.scene["cube_palm_contacts"].data.force_matrix_w[:, 0].norm(dim=-1)
                samples["palm_position_w"] = safe.palm_reference_pose_w()[0]
                samples["arm_effort_nm"] = env.action_manager.get_term("arm_action").joint_efforts
                samples["arm_torque_saturated"] = env.action_manager.get_term("arm_action").torque_saturated
                for name, sample in samples.items():
                    if name not in recorded:
                        recorded[name] = sample.clone()
                    recorded[name][live] = sample[live]
                return value

            def diagnostic(label):
                print("PUSH_PHYSICS_FIXTURE", {"mode": "prescribed robot geometry" if args.prescribed_robot_fixture else "OSC actions", "label": label, **{name: value.tolist() for name, value in recorded.items()}}, flush=True)

            def prescribed_step(desired_position):
                """Advance real physics with a separately prescribed robot path.

                This intentionally bypasses actuator strength and never
                supplies Cube forces or fabricated sensor/history values.
                Its IK and geometry checks belong only to this test fixture.
                """
                nonlocal fixture_ik_iterations, fixture_obb_checks
                identity = torch.eye(6, device=env.device).expand(env.num_envs, -1, -1)
                for _ in range(12):
                    position, quaternion = control_point_pose_w(env)
                    position_error, rotation_error = compute_pose_error(position, quaternion, desired_position, initial_quaternion, rot_error_type="axis_angle")
                    if bool(((position_error.norm(dim=-1) <= 1e-4) & (rotation_error.norm(dim=-1) <= 2e-3)).all()):
                        break
                    jacobian = safe._numerical_control_jacobian(all_ids)
                    error = torch.cat((position_error, rotation_error), -1)
                    solve = torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + .02**2 * identity, error[..., None])
                    delta = (.8 * (jacobian.transpose(1, 2) @ solve).squeeze(-1)).clamp(-.03, .03)
                    q = robot.data.joint_pos.clone()
                    q[:, safe.arm_joint_ids] = safe._bounded_arm(q[:, safe.arm_joint_ids] + delta, all_ids)
                    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
                    fixture_ik_iterations += 1
                position, quaternion = control_point_pose_w(env)
                position_error, rotation_error = compute_pose_error(position, quaternion, desired_position, initial_quaternion, rot_error_type="axis_angle")
                assert bool((position_error.norm(dim=-1) < 5e-4).all()), "Prescribed fixture IK did not converge"
                assert bool((rotation_error.norm(dim=-1) < .005).all()), "Prescribed fixture lost horizontal palm orientation"
                table = env.scene["table"]
                overlap = safe.collision_bounds.overlaps(robot.data.body_link_pos_w, robot.data.body_link_quat_w, table.data.root_pos_w, table.data.root_quat_w, cfg.task.table_size, cfg.task.reset_clearance_margin_m)
                fixture_obb_checks += 1
                assert not bool(overlap.any()), "Prescribed fixture path violated all-robot Table clearance"
                prescribed_q = robot.data.joint_pos.clone()
                for _ in range(env.cfg.decimation):
                    zero = torch.zeros_like(prescribed_q)
                    robot.write_joint_state_to_sim(prescribed_q, zero)
                    robot.set_joint_position_target(prescribed_q)
                    robot.set_joint_velocity_target(zero)
                    robot.set_joint_effort_target(zero)
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    env._sim_step_counter += 1
                env.episode_length_buf += 1
                value = command.state()
                reward = env.reward_manager.compute(dt=env.step_dt)
                finite("prescribed fixture reward", reward)
                assert not bool(value.failure.any()), "Prescribed fixture caused a physical failure"
                observation = env.observation_manager.compute()
                check_policy(observation)
                return observation, reward, value

            command.state = recording_state
            for index in range(args.physical_steps):
                # Scripted OSC translation is expressed through the measured
                # current C axes, with the inherited action scales and gains.
                _, quaternion = control_point_pose_w(env)
                world_delta = .012 * direction
                world_delta[ever_push & (peak_forward >= .015)] = 0.
                palm_position = safe.palm_reference_pose_w()[0]
                world_delta[:, 2] = (initial_palm_height - palm_position[:, 2]).clamp(-.004, .004)
                action[:, :3] = (quat_apply_inverse(quaternion, world_delta) / translation_scale).clamp(-1., 1.)
                orientation_error = quat_mul(quat_conjugate(quaternion), initial_quaternion)
                action[:, 3:6] = (axis_angle_from_quat(orientation_error) / rotation_scale).clamp(-1., 1.)
                try:
                    if args.prescribed_robot_fixture:
                        prescribed_distance += .001 * (~(ever_push & (peak_forward >= .015))).float()
                        observation, _, state = prescribed_step(initial_control_position + prescribed_distance[:, None] * direction)
                    else:
                        observation, _, state = step(action)
                except AssertionError:
                    diagnostic(f"push_failure_step_{index}")
                    command.state = original_state
                    raise
                ever_contact |= state.valid_push_contact
                ever_push |= state.push_seen
                forward = ((cube.data.root_pos_w - physical_initial) * direction).sum(-1)
                peak_forward = torch.maximum(peak_forward, forward)
                if args.verbose and index % 25 == 0:
                    diagnostic(f"push_step_{index}")
                if bool((ever_push & (peak_forward >= .015)).all()):
                    break
            diagnostic("push_stopped")
            assert bool(ever_contact.all()), f"Physical fixture did not produce supported side contact: {ever_contact.tolist()}"
            assert bool(ever_push.all()), f"Physical fixture produced no contact-qualified displacement: {peak_forward.tolist()}"
            # Stop applying forward deltas and let the dynamic Cube settle.
            action[:, :6] = 0.
            if args.prescribed_robot_fixture:
                # Release the imposed contact before measuring free settling.
                # Height/orientation stay fixed and Cube physics stays unforced.
                prescribed_distance = (prescribed_distance - .006).clamp_min(0.)
            settled = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            for _ in range(35):
                _, quaternion = control_point_pose_w(env)
                palm_position = safe.palm_reference_pose_w()[0]
                vertical_delta = torch.zeros_like(direction)
                vertical_delta[:, 2] = (initial_palm_height - palm_position[:, 2]).clamp(-.004, .004)
                action[:, :3] = (quat_apply_inverse(quaternion, vertical_delta) / translation_scale).clamp(-1., 1.)
                orientation_error = quat_mul(quat_conjugate(quaternion), initial_quaternion)
                action[:, 3:6] = (axis_angle_from_quat(orientation_error) / rotation_scale).clamp(-1., 1.)
                try:
                    if args.prescribed_robot_fixture:
                        observation, _, state = prescribed_step(initial_control_position + prescribed_distance[:, None] * direction)
                    else:
                        observation, _, state = step(action)
                except AssertionError:
                    diagnostic("settle_failure")
                    command.state = original_state
                    raise
                speed = cube.data.root_lin_vel_w.norm(dim=-1)
                stable = state.grounded & (speed <= cfg.task.success_speed_m_s)
                settled = torch.where(stable, settled + 1, torch.zeros_like(settled))
            assert bool((settled >= math.ceil(cfg.task.success_hold_time_s / env.step_dt)).all()), f"Dynamic Cube failed to settle after scripted stop: {settled.tolist()}"
            same_counters(instrumentation)
            diagnostic("settled")
            command.state = original_state
            fixture_result = {"mode": "prescribed robot geometry" if args.prescribed_robot_fixture else "OSC actions", "steps": index + 1, "supported_side_contact": ever_contact.tolist(), "push_seen": ever_push.tolist(), "forward_displacement_m": peak_forward.tolist(), "settled_steps": settled.tolist(),
                              "fixture_ik_iterations": fixture_ik_iterations, "fixture_obb_checks": fixture_obb_checks}
            print("[PASS] Scripted physical side-contact → Cube displacement → stop/settle fixture", fixture_result, flush=True)

        print("[PASS] Push fixture" if args.physical_only else "[PASS] Push 61D manager/reset/sensor/command contracts", {
            "boundary_specs": 0 if args.physical_only else len(corners) * len(angles),
            "minimum_distance_specs": 0 if args.physical_only else 2,
            "random_resets": 0 if args.physical_only else args.reset_samples,
            "timeout_checked": args.check_timeout and not args.physical_only,
            "reset_seconds": reset_seconds, "checked_step_seconds_mean": None if args.physical_only else step_seconds,
            "tracking_position_error_m": peak_tracking_position.tolist(), "tracking_angle_error_rad": peak_tracking_angle.tolist(),
            "physical_fixture": fixture_result,
        }, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
