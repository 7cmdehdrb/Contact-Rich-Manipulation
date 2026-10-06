#!/usr/bin/env python3
"""Check single-object resets, right-facing fixed OSC, relative observations and sensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

TASK_ID = "Isaac-Sweep-Inspire-Right-OSC-v0"
OBJECT_NAMES = ("bottle_1", "cup_1", "cup_2", "mug_1", "mug_2", "can_1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(TASK_ID,), default=TASK_ID)
parser.add_argument("--object-name", choices=OBJECT_NAMES, default="cup_1")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--debug-vis", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

from sweep_inspire_rl import TASK_ID as REGISTERED_TASK_ID  # noqa: E402
from sweep_inspire_rl.env_cfg import InspireShelfSweepEnvCfg, POLICY_OBSERVATION_DIM  # noqa: E402
from sweep_inspire_rl.mdp.actions import bounded_hand_openness  # noqa: E402
from sweep_inspire_rl.mdp.observations import palm_tactile_bits, wrist_wrench_c  # noqa: E402


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains non-finite values")


def _check_relative_observations(env, policy: torch.Tensor) -> None:
    eef = env.scene["ee_frame"].data
    eef_position = eef.target_pos_w[:, 0, :]
    eef_quaternion = eef.target_quat_w[:, 0, :]
    target_position = env.scene["object_collection"].data.object_state_w[:, 0, :3]
    goal_position = env.command_manager.get_command("target_goal_pos")[:, :3]
    target_relative = math_utils.quat_apply_inverse(eef_quaternion, target_position - eef_position)
    goal_relative = math_utils.quat_apply_inverse(eef_quaternion, goal_position - eef_position)
    # arm+hand positions18, arm velocities6, last actions8 precede target;
    # target3, width1 and root-frame EEF pose7 precede the goal.
    torch.testing.assert_close(policy[:, 32:35], target_relative, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(policy[:, 43:46], goal_relative, rtol=1.0e-5, atol=1.0e-5)


def _check_right_command(env) -> None:
    expected = torch.zeros_like(env.sweep_dir)
    expected[:, 1] = 0.18
    torch.testing.assert_close(env.sweep_dir, expected, rtol=0.0, atol=1.0e-6)
    command = env.command_manager.get_term("target_goal_pos")
    torch.testing.assert_close(
        command.command[:, :3] - command.target_init_state_w[:, :3], expected,
        rtol=0.0, atol=1.0e-6,
    )


def _check_sensor_observations(env, policy: torch.Tensor) -> None:
    tactile = palm_tactile_bits(env)
    wrench = wrist_wrench_c(env)
    if tactile.shape != (env.num_envs, 17) or wrench.shape != (env.num_envs, 6):
        raise RuntimeError(f"Unexpected tactile/F/T shapes: {tactile.shape}, {wrench.shape}")
    _finite("tactile", tactile)
    _finite("F/T", wrench)
    torch.testing.assert_close(policy[:, 48:65], tactile)
    torch.testing.assert_close(policy[:, 65:71], wrench)


def main() -> None:
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    cfg = InspireShelfSweepEnvCfg(object_name=args.object_name)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    cfg.commands.target_goal_pos.debug_vis = args.debug_vis
    osc_cfg = cfg.actions.arm_action.controller_cfg
    if osc_cfg.impedance_mode != "fixed" or tuple(osc_cfg.target_types) != ("pose_rel",):
        raise RuntimeError("OSC must use pose_rel with fixed impedance")
    if tuple(osc_cfg.motion_stiffness_task) != (200.0,) * 6:
        raise RuntimeError("OSC stiffness must be 200 on all six axes")
    if len(cfg.scene.object_collection.rigid_objects) != 1:
        raise RuntimeError("Configuration must spawn exactly one object")

    wrapped_env = gym.make(REGISTERED_TASK_ID, cfg=cfg)
    env = wrapped_env.unwrapped
    try:
        observation, _ = env.reset()
        policy = observation["policy"]
        if policy.shape != (args.num_envs, POLICY_OBSERVATION_DIM):
            raise RuntimeError(f"Expected {POLICY_OBSERVATION_DIM}D observations, got {tuple(policy.shape)}")
        if env.action_manager.total_action_dim != 8:
            raise RuntimeError("Expected six OSC actions plus two Hand actions")
        if env.scene["object_collection"].num_objects != 1:
            raise RuntimeError("Runtime collection contains more than one object")
        if not bool(env.reaching_ik_success.all()):
            raise RuntimeError("Reset IK failed")
        if env.desired_reaching_pose_w.shape != (args.num_envs, 7):
            raise RuntimeError("Reset pose diagnostics have the wrong shape")
        hand = env.action_manager.get_term("hand_action")
        if hand.action_dim != 2 or hand.joint_targets.shape != (args.num_envs, 12):
            raise RuntimeError("Hand action must expand two synergy values into twelve joint targets")
        if bool((hand.actual_synergy < 0.8 - 0.02).any()) or bool((hand.actual_synergy > 1.0 + 0.02).any()):
            raise RuntimeError("Reset hand is outside the nearly open posture")
        torch.testing.assert_close(
            bounded_hand_openness(torch.tensor([[0.0, 1.0]], device=env.device)),
            torch.tensor([[0.8, 1.0]], device=env.device),
        )
        if not bool((env.single_action_space.low[-2:] == 0.0).all()):
            raise RuntimeError("Hand action lower limits must be zero")
        if not bool((env.single_action_space.high[-2:] == 1.0).all()):
            raise RuntimeError("Hand action upper limits must be one")
        _finite("reset observation", policy)
        _check_right_command(env)
        _check_relative_observations(env, policy)
        _check_sensor_observations(env, policy)

        eef_position = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        target_position = env.scene["object_collection"].data.object_state_w[:, 0, :3]
        if not bool((eef_position[:, 1] < target_position[:, 1]).all()):
            raise RuntimeError("Reset hand must start on the left of the target for a rightward sweep")
        if not bool((eef_position[:, 2] > target_position[:, 2]).all()):
            raise RuntimeError("Reset control point must start above the target's reference height")

        arm = env.action_manager.get_term("arm_action")
        zero_arm = torch.zeros((args.num_envs, 6), device=env.device)
        arm.process_actions(zero_arm)
        held_rotation = arm.processed_actions[:, 3:].clone()
        rotation_input = zero_arm.clone()
        rotation_input[:, 3:] = torch.tensor((0.5, -0.5, 1.0), device=env.device)
        arm.process_actions(rotation_input)
        torch.testing.assert_close(arm.processed_actions[:, 3:], held_rotation, rtol=0.0, atol=1.0e-6)

        action = torch.zeros((args.num_envs, 8), device=env.device)
        action[:, 6:] = 0.5
        for _ in range(args.steps):
            observation, reward, terminated, truncated, _ = env.step(action)
            policy = observation["policy"]
            _finite("step observation", policy)
            _finite("reward", reward)
            _finite("Hand joint targets", hand.joint_targets)
            if bool((hand.synergy_target < 0.8 - 1.0e-6).any()) or bool((hand.synergy_target > 1.0 + 1.0e-6).any()):
                raise RuntimeError("Hand controller issued a target outside [0.8, 1.0]")
            _check_right_command(env)
            _check_relative_observations(env, policy)
            _check_sensor_observations(env, policy)
            if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
                raise RuntimeError("Termination buffers must be boolean")
        env.reset()
        _check_right_command(env)
        if not bool(env.reaching_ik_success.all()):
            raise RuntimeError("Second reset IK failed")
        print(
            f"[PASS] {args.steps} finite steps; single {args.object_name}; "
            f"relative obs={POLICY_OBSERVATION_DIM}, actions=8, right=+Y, "
            "OSC stiffness=200 fixed, Hand openness=[0.8,1.0], tactile=17, F/T=6",
            flush=True,
        )
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
