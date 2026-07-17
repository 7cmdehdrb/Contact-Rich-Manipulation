#!/usr/bin/env python3
"""Play a trained UR5e OSC sweep policy with task-specific visualization.

The goal object pose is drawn as a coordinate frame. Desired and measured
contact forces are drawn as green and red arrows respectively. A compact log
for environment zero is also printed periodically, including in headless mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play and visualize a trained OSC sweep policy.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Sweep-Object-UR5e-OSC-Play-v0",
    help="Gym task ID. The Play variant uses deterministic sweep commands.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model_*.pt checkpoint.")
parser.add_argument(
    "--load_run",
    type=str,
    default=".*",
    help="Run directory name or regex used when --checkpoint is omitted.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed override.")
parser.add_argument(
    "--log_interval",
    type=int,
    default=30,
    help="Print force and goal data every N policy steps; zero disables logging.",
)
parser.add_argument(
    "--force_scale",
    type=float,
    default=0.01,
    help="Force-arrow length in meters per newton.",
)
parser.add_argument("--real_time", action="store_true", help="Throttle playback to simulated real time.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Stop after N policy steps; zero runs until the simulator closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Hydra should only see task-configuration overrides.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401
import sweep_rl  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from sweep_rl.osc_sweep.mdp.common import target_contact_data_w


RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def _force_marker_cfg(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Create an X-aligned force arrow with the requested color."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def _vector_to_arrow(vector_w: torch.Tensor, meters_per_unit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert world vectors to X-axis arrow orientations and length scales."""
    magnitude = torch.linalg.norm(vector_w, dim=-1)
    safe_magnitude = torch.clamp(magnitude, min=1.0e-6)
    x_axis = vector_w / safe_magnitude.unsqueeze(-1)

    fallback_x = torch.zeros_like(x_axis)
    fallback_x[:, 0] = 1.0
    x_axis = torch.where((magnitude > 1.0e-6).unsqueeze(-1), x_axis, fallback_x)

    reference_z = torch.zeros_like(x_axis)
    reference_z[:, 2] = 1.0
    reference_y = torch.zeros_like(x_axis)
    reference_y[:, 1] = 1.0
    reference = torch.where((torch.abs(x_axis[:, 2]) > 0.95).unsqueeze(-1), reference_y, reference_z)

    y_axis = torch.nn.functional.normalize(torch.cross(reference, x_axis, dim=-1), dim=-1)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    rotation_w = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    orientation_w = math_utils.quat_from_matrix(rotation_w)

    scales = torch.ones(vector_w.shape[0], 3, device=vector_w.device)
    scales[:, 0] = torch.clamp(magnitude * meters_per_unit, min=1.0e-4)
    scales[:, 1:] = 0.5
    return orientation_w, scales


class SweepVisualizer:
    """Own and update the goal, desired-force, and measured-force markers."""

    def __init__(self, env, force_scale: float):
        self.env = env
        self.force_scale = force_scale
        self.command = env.command_manager.get_term("desired_motion")
        self.target = env.scene[self.command.cfg.object_name]
        self.robot = env.scene[self.command.cfg.robot_name]

        goal_cfg = FRAME_MARKER_CFG.copy()
        goal_cfg.prim_path = "/Visuals/Sweep/GoalPose"
        goal_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.goal_marker = VisualizationMarkers(goal_cfg)
        self.desired_force_marker = VisualizationMarkers(
            _force_marker_cfg("/Visuals/Sweep/DesiredForce", (0.0, 1.0, 0.0))
        )
        self.measured_force_marker = VisualizationMarkers(
            _force_marker_cfg("/Visuals/Sweep/MeasuredForce", (1.0, 0.0, 0.0))
        )

    def update(self) -> dict[str, torch.Tensor]:
        command = self.command.command
        goal_pos_w = self.command.goal_pos_w

        initial_rpy_b = self.command.initial_pose_b[:, 3:6]
        initial_quat_b = math_utils.quat_from_euler_xyz(
            initial_rpy_b[:, 0], initial_rpy_b[:, 1], initial_rpy_b[:, 2]
        )
        goal_quat_w = math_utils.quat_mul(self.robot.data.root_quat_w, initial_quat_b)
        self.goal_marker.visualize(goal_pos_w, goal_quat_w)

        contact_pos_w, sensor_force_w, contact_mask = target_contact_data_w(
            self.env, self.command.cfg.contact_sensor_names
        )
        # ContactSensor reports force on the gripper pad. Negating it gives the
        # corresponding force exerted by the pad on the target object.
        measured_push_force_w = -sensor_force_w
        desired_force_w = self.command.direction_w * command[:, 3:4]

        object_pos_w = self.target.data.root_pos_w
        desired_origin_w = object_pos_w.clone()
        desired_origin_w[:, 2] += 0.10
        measured_origin_w = torch.where(contact_mask.unsqueeze(-1), contact_pos_w, object_pos_w).clone()
        measured_origin_w[:, 2] += 0.13

        desired_quat_w, desired_scale = _vector_to_arrow(desired_force_w, self.force_scale)
        measured_quat_w, measured_scale = _vector_to_arrow(measured_push_force_w, self.force_scale)
        self.desired_force_marker.visualize(desired_origin_w, desired_quat_w, desired_scale)
        self.measured_force_marker.visualize(measured_origin_w, measured_quat_w, measured_scale)

        return {
            "goal_pos_w": goal_pos_w,
            "object_pos_w": object_pos_w,
            "desired_force": command[:, 3],
            "force_tolerance": command[:, 4],
            "measured_force": torch.linalg.norm(measured_push_force_w, dim=-1),
            "contact_mask": contact_mask,
        }


def _print_status(step: int, values: dict[str, torch.Tensor]):
    """Print a compact status line for environment zero."""
    index = 0
    goal = values["goal_pos_w"][index].tolist()
    current = values["object_pos_w"][index].tolist()
    goal_error = torch.linalg.norm(values["goal_pos_w"][index] - values["object_pos_w"][index]).item()
    print(
        f"[SWEEP][step={step:06d}][env=0] "
        f"goal=({goal[0]:+.3f}, {goal[1]:+.3f}, {goal[2]:+.3f}) m | "
        f"object=({current[0]:+.3f}, {current[1]:+.3f}, {current[2]:+.3f}) m | "
        f"error={goal_error:.3f} m | "
        f"force desired={values['desired_force'][index].item():.2f} N, "
        f"measured={values['measured_force'][index].item():.2f} N, "
        f"tolerance=+/-{values['force_tolerance'][index].item():.2f} N | "
        f"contact={'yes' if values['contact_mask'][index].item() else 'no'}",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Load a checkpoint and run sweep-policy inference."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    agent_cfg.device = args_cli.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint_path = get_checkpoint_path(log_root, args_cli.load_run, "model_.*.pt")
    print(f"[INFO] Loading sweep policy: {checkpoint_path}")
    print("[INFO] Markers: goal pose=frame, desired force=green, measured force=red")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    visualizer = SweepVisualizer(base_env, args_cli.force_scale)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = os.path.dirname(checkpoint_path)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    checkpoint_path = handle_deprecated_rsl_rl_checkpoint(checkpoint_path, RSL_RL_VERSION)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=base_env.device)

    obs = env.get_observations()
    step = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    runner.alg.actor_critic.reset(dones)
                values = visualizer.update()

            if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
                _print_status(step, values)
            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

            sleep_time = base_env.step_dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
