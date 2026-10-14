#!/usr/bin/env python3
"""Probe six C-frame axes against paired zero-action environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--amplitude", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--surface", choices=("palm", "dorsal"), default="palm")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

from hand_manipulation_rl import TASK_ID  # noqa: E402
from hand_manipulation_rl.env_cfg import BlindSweepEnvCfg  # noqa: E402


AXIS_NAMES = ("+X", "+Y", "+Z", "+Rx", "+Ry", "+Rz")


def main() -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if not 0.0 < args.amplitude <= 0.5:
        raise ValueError("--amplitude must lie in (0, 0.5]")

    cfg = BlindSweepEnvCfg()
    # The reference OSC deliberately omits gravity compensation.  Pair every
    # commanded environment with an identical zero-action environment so the
    # measured response subtracts the common gravity-driven motion.
    cfg.scene.num_envs = 12
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.task.palm_mode_probability = 1.0 if args.surface == "palm" else 0.0
    env = gym.make(TASK_ID, cfg=cfg).unwrapped
    observation, _ = env.reset()
    probe_ids = torch.arange(6, device=env.device)
    baseline_ids = probe_ids + 6

    robot = env.scene["robot"]
    paired_joint_pos = robot.data.joint_pos[probe_ids].clone()
    robot.write_joint_state_to_sim(
        paired_joint_pos,
        torch.zeros_like(paired_joint_pos),
        env_ids=baseline_ids,
    )
    target = env.scene["target_object"]
    paired_target_pos_local = (
        target.data.root_pos_w[probe_ids] - env.scene.env_origins[probe_ids]
    )
    paired_target_pose = torch.cat(
        (
            paired_target_pos_local + env.scene.env_origins[baseline_ids],
            target.data.root_quat_w[probe_ids].clone(),
        ),
        dim=-1,
    )
    target.write_root_pose_to_sim(paired_target_pose, env_ids=baseline_ids)
    target.write_root_velocity_to_sim(
        torch.zeros((6, 6), device=env.device), env_ids=baseline_ids
    )
    env.sim.forward()
    env.scene.update(dt=0.0)
    for term in env.action_manager._terms.values():
        reset_to_current = getattr(term, "reset_to_current", None)
        if reset_to_current is not None:
            reset_to_current(torch.arange(12, device=env.device))

    initial_position_w, initial_quaternion_w = env.control_point_pose_w()
    initial_position_w = initial_position_w.clone()
    initial_quaternion_w = initial_quaternion_w.clone()

    action = torch.zeros((12, 8), device=env.device)
    action[probe_ids, probe_ids] = args.amplitude
    actual_synergy = env.action_manager.get_term("hand_action").actual_synergy
    action[:, 6:] = 2.0 * actual_synergy - 1.0

    for _ in range(args.steps):
        observation, reward, terminated, truncated, extras = env.step(action)
        diagnostics = extras["episode_diagnostics"]
        if not torch.isfinite(observation["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("Axis probe produced non-finite observations or rewards")
        if bool(diagnostics["arm_torque_saturated"].any()):
            raise RuntimeError("A small C-frame increment saturated an arm torque")
        if bool((terminated | truncated).any()):
            raise RuntimeError(
                "Axis probe terminated unexpectedly: "
                f"reason={extras['termination_reason'].tolist()}"
            )

    final_position_w, final_quaternion_w = env.control_point_pose_w()
    local_translation = math_utils.quat_apply_inverse(
        initial_quaternion_w, final_position_w - initial_position_w
    )
    relative_quaternion = math_utils.quat_mul(
        math_utils.quat_inv(initial_quaternion_w), final_quaternion_w
    )
    local_rotation = math_utils.axis_angle_from_quat(relative_quaternion)
    desired_quaternion_w = diagnostics["desired_c_quaternion_w"]
    desired_relative_quaternion = math_utils.quat_mul(
        math_utils.quat_inv(initial_quaternion_w), desired_quaternion_w
    )
    desired_local_rotation = math_utils.axis_angle_from_quat(
        desired_relative_quaternion
    )
    remaining_relative_quaternion = math_utils.quat_mul(
        math_utils.quat_inv(final_quaternion_w), desired_quaternion_w
    )
    remaining_local_rotation = math_utils.axis_angle_from_quat(
        remaining_relative_quaternion
    )
    differential_translation = local_translation[probe_ids] - local_translation[baseline_ids]
    differential_rotation = local_rotation[probe_ids] - local_rotation[baseline_ids]
    translation_response = differential_translation[:3].diagonal()
    rotation_response = differential_rotation[3:].diagonal()

    print(f"surface: {args.surface}")
    print("axes:", AXIS_NAMES)
    print("local translation response [m]:", local_translation.tolist())
    print("local rotation response [rad]:", local_rotation.tolist())
    print("paired-baseline-subtracted translation response [m]:", differential_translation.tolist())
    print("paired-baseline-subtracted rotation response [rad]:", differential_rotation.tolist())
    print("desired local rotation [rad]:", desired_local_rotation.tolist())
    print("remaining local rotation error [rad]:", remaining_local_rotation.tolist())
    print("commanded translation components [m]:", translation_response.tolist())
    print("commanded rotation components [rad]:", rotation_response.tolist())
    print("final desired C quaternion W:", diagnostics["desired_c_quaternion_w"].tolist())
    print("final measured C twist B:", diagnostics["measured_c_twist_b"].tolist())
    print("final arm effort [Nm]:", diagnostics["arm_joint_effort_nm"].tolist())
    sys.stdout.flush()

    if bool((translation_response <= 5.0e-5).any()):
        raise RuntimeError(
            f"C-frame translation axis did not move in its commanded direction: "
            f"{translation_response.tolist()}"
        )
    if bool((rotation_response <= 1.0e-4).any()):
        raise RuntimeError(
            f"C-frame rotation axis did not move in its commanded direction: "
            f"{rotation_response.tolist()}"
        )
    if bool((torch.linalg.vector_norm(differential_translation[:3], dim=-1) > 0.05).any()):
        raise RuntimeError("Small translation probe produced an implausibly large displacement")
    if bool((torch.linalg.vector_norm(differential_rotation[3:], dim=-1) > 0.25).any()):
        raise RuntimeError("Small rotation probe produced an implausibly large rotation")
    if bool((env.board_force_peak > 0.0).any()):
        raise RuntimeError(f"Axis probe touched the board: {env.board_force_peak.tolist()} N")

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
        import omni.kit.app

        omni.kit.app.get_app().post_quit(exit_code)
        sys.stdout.flush()
        sys.stderr.flush()
        if args.headless and args.device == "cpu":
            os._exit(exit_code)
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(exit_code)
