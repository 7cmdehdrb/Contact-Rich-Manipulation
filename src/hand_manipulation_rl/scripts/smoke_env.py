#!/usr/bin/env python3
"""Validate resets, reference OSC dynamics, and the actual hand posture."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--trace",
    action="store_true",
    help="Print per-policy-step uncompensated OSC dynamics diagnostics.",
)
parser.add_argument(
    "--surface",
    choices=("both", "palm", "dorsal"),
    default="both",
    help="Restrict the reset sampler while diagnosing one surface mode.",
)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    help="Create and update Target, Goal, and virtual EEF markers during the smoke run.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

from hand_manipulation_rl import TASK_ID  # noqa: E402
from hand_manipulation_rl.assets.robot import (  # noqa: E402
    FT_SENSOR_BODY_NAME,
    ROBOT_CONTACT_BODY_NAMES,
)
from hand_manipulation_rl.env_cfg import BlindSweepEnvCfg  # noqa: E402


def main() -> None:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    cfg = BlindSweepEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = args.debug_vis
    if args.surface != "both":
        cfg.task.palm_mode_probability = 1.0 if args.surface == "palm" else 0.0
    env = gym.make(TASK_ID, cfg=cfg).unwrapped
    observation, _ = env.reset()
    if args.debug_vis:
        if env._task_point_markers is None or env._eef_frame_markers is None:
            raise RuntimeError("Debug visualization markers were not created")
        if env._task_point_markers.count != 2 * args.num_envs:
            raise RuntimeError("Target/Goal marker instance count is incorrect")
        if env._eef_frame_markers.count != 4 * args.num_envs:
            raise RuntimeError("Virtual EEF marker instance count is incorrect")
    policy = observation["policy"]
    if policy.shape != (args.num_envs, 57):
        raise RuntimeError(
            f"Expected a ({args.num_envs}, 57) policy observation, got {tuple(policy.shape)}"
        )
    if env.action_manager.total_action_dim != 8:
        raise RuntimeError(f"Expected 8 actions, got {env.action_manager.total_action_dim}")
    if not bool(env.reset_ik_success.all()):
        raise RuntimeError("Initial full-pose IK reset did not succeed")
    if not bool(env.reset_collision_free.all()):
        raise RuntimeError("Initial reset did not pass live-FK collision certification")
    if not bool(env.sensor_valid.all()) or bool(env.sensor_data_fresh.any()):
        raise RuntimeError(
            "Reset must expose a valid initialized-zero sensor buffer before live physics data"
        )
    print("robot joints:", env.scene["robot"].joint_names)
    print("robot bodies:", env.scene["robot"].body_names)
    print(
        "reset mode/theta/distance:",
        env.surface_mode.tolist(),
        env.command_angle.tolist(),
        env.command_distance.tolist(),
    )
    expected_direction = torch.zeros_like(env.command_direction_w)
    expected_direction[:, 1] = torch.sign(env.command_direction_w[:, 1])
    torch.testing.assert_close(env.command_direction_w, expected_direction, rtol=0.0, atol=0.0)
    if not bool((torch.abs(env.command_direction_w[:, 1]) == 1.0).all()):
        raise RuntimeError("Sweep command must be exactly shelf-left or shelf-right along world Y")
    torch.testing.assert_close(
        env.command_distance,
        torch.full_like(env.command_distance, 0.18),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        env.goal_pos_w - env.object_initial_pos_w,
        0.18 * env.command_direction_w,
        rtol=0.0,
        atol=1.0e-6,
    )
    initial_c_position_w, initial_c_quaternion_w = env.control_point_pose_w()
    initial_c_position_w = initial_c_position_w.clone()
    initial_c_quaternion_w = initial_c_quaternion_w.clone()
    action = torch.zeros((args.num_envs, 8), device=env.device)
    jacobian_twist_error_peak = torch.zeros(args.num_envs, device=env.device)
    physical_contact_peak = torch.zeros(args.num_envs, device=env.device)
    # Policy hand inputs use [-1, 1]; keep the actual reset posture rather than
    # commanding a sudden close in this dynamics smoke test.
    action[:, 6:] = 2.0 * env.initial_hand_actual_synergy - 1.0
    for step_index in range(args.steps):
        observation, reward, terminated, truncated, extras = env.step(action)
        episode_diagnostics = extras["episode_diagnostics"]
        robot_contact = torch.linalg.vector_norm(
            env.scene["robot_contacts"].data.net_forces_w, dim=-1
        ).amax(dim=-1)
        target_matrix = env.scene["target_robot_contacts"].data.force_matrix_w
        if target_matrix is None:
            raise RuntimeError("Target reset-contact sensor has no force matrix")
        target_contact = torch.linalg.vector_norm(target_matrix, dim=-1).flatten(1).amax(dim=-1)
        physical_contact_peak = torch.maximum(
            physical_contact_peak,
            torch.maximum(robot_contact, torch.maximum(target_contact, env.board_force)),
        )
        if args.trace:
            print(
                f"step {step_index + 1} OSC diagnostics:",
                {
                    "c_position_w": episode_diagnostics["c_position_w"].tolist(),
                    "c_quaternion_w": episode_diagnostics["c_quaternion_w"].tolist(),
                    "desired_c_position_w": episode_diagnostics[
                        "desired_c_position_w"
                    ].tolist(),
                    "desired_c_quaternion_w": episode_diagnostics[
                        "desired_c_quaternion_w"
                    ].tolist(),
                    "measured_c_twist_b": episode_diagnostics[
                        "measured_c_twist_b"
                    ].tolist(),
                    "jacobian_c_twist_at_control_b": episode_diagnostics[
                        "jacobian_c_twist_at_control_b"
                    ].tolist(),
                    "arm_joint_effort_nm": episode_diagnostics[
                        "arm_joint_effort_nm"
                    ].tolist(),
                    "arm_torque_saturated": episode_diagnostics[
                        "arm_torque_saturated"
                    ].tolist(),
                    "arm_joint_velocity_rad_s": episode_diagnostics[
                        "arm_joint_velocity_rad_s"
                    ].tolist(),
                },
                flush=True,
            )
        if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("Smoke run produced non-finite observations/rewards")
        if bool(episode_diagnostics["arm_torque_saturated"].any()):
            raise RuntimeError("Zero-arm-delta OSC unexpectedly saturated an arm torque")
        jacobian_twist_error = torch.linalg.vector_norm(
            episode_diagnostics["jacobian_c_twist_at_control_b"]
            - episode_diagnostics["measured_c_twist_b"],
            dim=-1,
        )
        jacobian_twist_error_peak = torch.maximum(
            jacobian_twist_error_peak, jacobian_twist_error
        )
        # At the reference 100 Hz physics rate, CPU/USD fallback updates the
        # link-twist tensor one solver sample apart from the control Jacobian.
        # The resulting sub-mm/s discrepancy is still a strict frame check;
        # allow up to 1 mm/s so a one-sample CPU fallback skew does not fail.
        if bool((jacobian_twist_error > 1.0e-3).any()):
            raise RuntimeError(
                "C Jacobian and measured C twist disagree: "
                f"{jacobian_twist_error.tolist()}"
            )
        if bool((terminated | truncated).any()):
            details = (
                f"terminated={terminated.tolist()}, truncated={truncated.tolist()}, "
                f"reason={extras['termination_reason'].tolist()}"
            )
            print(f"uncompensated zero-delta smoke termination: {details}", flush=True)
            board_link_indices = episode_diagnostics["board_link_index"].tolist()
            board_link_names = [
                ROBOT_CONTACT_BODY_NAMES[index] for index in board_link_indices
            ]
            print(
                "episode termination diagnostics:",
                {
                    "board_force_peak_n": episode_diagnostics["board_force_peak_n"].tolist(),
                    "board_link_force_n": episode_diagnostics["board_link_force_n"].tolist(),
                    "board_link": board_link_names,
                    "board_link_position_w": episode_diagnostics[
                        "board_link_position_w"
                    ].tolist(),
                    "c_position_at_board_peak_w": episode_diagnostics[
                        "board_contact_c_position_w"
                    ].tolist(),
                    "tilt_peak_rad": episode_diagnostics["tilt_peak_rad"].tolist(),
                    "height_peak_m": episode_diagnostics["height_peak_m"].tolist(),
                    "c_position_w": episode_diagnostics["c_position_w"].tolist(),
                    "desired_c_position_w": episode_diagnostics[
                        "desired_c_position_w"
                    ].tolist(),
                    "arm_joint_effort_nm": episode_diagnostics[
                        "arm_joint_effort_nm"
                    ].tolist(),
                    "arm_torque_saturated": episode_diagnostics[
                        "arm_torque_saturated"
                    ].tolist(),
                    "arm_joint_velocity_rad_s": episode_diagnostics[
                        "arm_joint_velocity_rad_s"
                    ].tolist(),
                },
                flush=True,
            )
            print(
                "reset validation diagnostics:",
                {
                    "collision_free": episode_diagnostics[
                        "reset_collision_free"
                    ].tolist(),
                    "zero_action_physical_contact_peak_n": physical_contact_peak.tolist(),
                    "conservative_min_clearance_m": episode_diagnostics[
                        "reset_min_clearance_m"
                    ].tolist(),
                },
                flush=True,
            )
            raise RuntimeError(
                "Uncompensated zero-delta smoke reached a task termination; "
                f"use a shorter reset probe or command active support: {details}"
            )
    if not bool(env.sensor_valid.all()):
        raise RuntimeError("Wrench sensor initialization did not complete")
    wrench_sample = env.wrist_wrench_c()
    measured_wrench = wrench_sample.measured_c
    if not torch.isfinite(measured_wrench).all():
        raise RuntimeError("Measured C-frame wrench is non-finite")
    torch.testing.assert_close(wrench_sample.measured_f, wrench_sample.raw_f)
    if bool((torch.linalg.vector_norm(wrench_sample.raw_f[:, :3], dim=-1) <= 1.0e-4).any()):
        raise RuntimeError("Uncancelled F/T sensor did not retain the Hand self-weight force")
    if bool((torch.linalg.vector_norm(wrench_sample.raw_f[:, 3:], dim=-1) <= 1.0e-6).any()):
        raise RuntimeError("Uncancelled F/T sensor did not retain the Hand self-weight moment")
    final_c_position_w, final_c_quaternion_w = env.control_point_pose_w()
    hold_position_drift_m = torch.linalg.vector_norm(
        final_c_position_w - initial_c_position_w, dim=-1
    )
    quaternion_dot = torch.abs(
        torch.sum(final_c_quaternion_w * initial_c_quaternion_w, dim=-1)
    ).clamp(max=1.0)
    hold_orientation_drift_rad = 2.0 * torch.acos(quaternion_dot)
    # The Sweep-Policy reference deliberately disables OSC gravity
    # compensation.  A zero relative command therefore damps motion but is
    # not a static hold command under the UR5e + Hand payload.  Report that
    # physically expected drift instead of treating it as a controller error.
    if bool((physical_contact_peak > env.cfg.task.reset_contact_tolerance_n).any()):
        raise RuntimeError(
            "Accepted reset produced a physical contact above tolerance: "
            f"{physical_contact_peak.tolist()} N"
        )
    print("policy shape:", tuple(observation["policy"].shape))
    print("uncompensated zero-delta C position drift [m]:", hold_position_drift_m.tolist())
    print("uncompensated zero-delta C orientation drift [rad]:", hold_orientation_drift_rad.tolist())
    print("zero-action physical contact peak [N]:", physical_contact_peak.tolist())
    print("C Jacobian/twist error peak:", jacobian_twist_error_peak.tolist())
    print("board peak [N]:", env.board_force_peak.tolist())
    print("wrench cancellation: disabled; Hand self-weight force/moment retained")
    print("raw wrench F:", wrench_sample.raw_f.tolist())
    print("measured wrench C:", measured_wrench.tolist())
    print("reset IK error [m, rad]:", env.reset_position_error.tolist(), env.reset_orientation_error.tolist())

    # Guard the procedural Axia80 extent regression explicitly.  Reading its
    # stale authored USD extent yields a one-metre half-size; geometry-plugin
    # evaluation must recover the configured cylinder dimensions instead.
    axia_ids, axia_names = env.scene["robot"].find_bodies(
        FT_SENSOR_BODY_NAME, preserve_order=True
    )
    if len(axia_ids) != 1 or axia_names[0] != FT_SENSOR_BODY_NAME:
        raise RuntimeError("Could not resolve Axia80 body for collision-bound validation")
    expected_axia_half_extent = torch.tensor(
        (
            cfg.scene.robot.spawn.sensor_radius,
            cfg.scene.robot.spawn.sensor_radius,
            0.5 * cfg.scene.robot.spawn.sensor_height,
        ),
        device=env.device,
    )
    actual_axia_half_extent = env._reset_collision_half_extents[int(axia_ids[0])]
    if not torch.allclose(
        actual_axia_half_extent, expected_axia_half_extent, atol=1.0e-6, rtol=0.0
    ):
        raise RuntimeError(
            "Axia80 collision extent does not match the configured cylinder: "
            f"actual={actual_axia_half_extent.tolist()}, "
            f"expected={expected_axia_half_extent.tolist()}"
        )

    # Negative controls for every robot body: place the live Cube at the
    # center of each conservative collision OBB and require that exact body to
    # be reported.  Tensor writes update the RigidObject buffer immediately,
    # so these checks advance no simulation time and expose no transition.
    robot = env.scene["robot"]
    target = env.scene["target_object"]
    all_env_ids = torch.arange(args.num_envs, dtype=torch.long, device=env.device)
    target_quaternion_w = target.data.root_quat_w.clone()
    target.write_root_velocity_to_sim(
        torch.zeros((args.num_envs, 6), device=env.device), env_ids=all_env_ids
    )
    for body_id, body_name in enumerate(robot.body_names):
        local_center = env._reset_collision_local_centers[body_id].expand(
            args.num_envs, -1
        )
        collision_center_w = robot.data.body_pos_w[:, body_id] + math_utils.quat_apply(
            robot.data.body_quat_w[:, body_id], local_center
        )
        target.write_root_pose_to_sim(
            torch.cat((collision_center_w, target_quaternion_w), dim=-1),
            env_ids=all_env_ids,
        )
        collision_probe_free = env.reset_collision_free_mask(all_env_ids)
        body_detected = env._reset_last_cube_overlap[:, body_id]
        if bool(collision_probe_free.any()) or not bool(body_detected.all()):
            raise RuntimeError(
                "Reset collision certificate missed a deliberate Robot--Cube overlap "
                f"on {body_name}: free={collision_probe_free.tolist()}, "
                f"body_detected={body_detected.tolist()}"
            )
    print(
        "deliberate Robot--Cube overlaps rejected:",
        f"{len(robot.body_names)} bodies x {args.num_envs} environments",
    )
    env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # Isaac Sim 5.1 can stall during exhaustive headless cleanup when no
        # graphics device exists.  Post the result before the standalone fast
        # shutdown path, which may terminate the process itself.
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac Sim 5.1 may segfault in renderer teardown on a headless host
        # with no graphics device. A diagnostic CPU process has no external
        # resources to preserve, so let the OS reclaim it after flushing.
        if args.headless and args.device == "cpu":
            os._exit(exit_code)
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)
