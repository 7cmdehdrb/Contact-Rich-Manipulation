#!/usr/bin/env python3
"""Play an RSL-RL sweep policy and visualize its object-position goal.

The marker reads ``goal_pos_w`` from the configured command term every step,
so it follows newly sampled goals when an episode resets.  It is visualization
only and does not change observations, actions, rewards, or termination logic.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Play an RSL-RL sweep policy with a visible object goal."
)
parser.add_argument("--task", type=str, required=True, help="Gym task ID.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model_*.pt checkpoint.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Gym registry key for the RSL-RL agent configuration.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to play.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed override.")
parser.add_argument(
    "--load_run",
    type=str,
    default=".*",
    help="Run directory name or regex used when --checkpoint is omitted.",
)
parser.add_argument(
    "--command_name",
    type=str,
    default="desired_motion",
    help="Command term that owns goal_pos_w.",
)
parser.add_argument(
    "--goal_radius",
    type=float,
    default=0.03,
    help="Radius of the green goal marker in meters.",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=60,
    help="Print goal status every N policy steps; zero disables logging.",
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

if args_cli.goal_radius <= 0.0:
    parser.error("--goal_radius must be positive")
if args_cli.log_interval < 0:
    parser.error("--log_interval cannot be negative")
if args_cli.max_steps < 0:
    parser.error("--max_steps cannot be negative")

# Only the unparsed arguments belong to Hydra.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401
import sweep_rl  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


RSL_RL_VERSION = metadata.version("rsl-rl-lib")


class GoalVisualizer:
    """Draw the command's world-frame goal and report its position error."""

    def __init__(self, env, command_name: str, marker_radius: float):
        self.command = env.command_manager.get_term(command_name)
        if not hasattr(self.command, "goal_pos_w"):
            raise AttributeError(
                f"Command term '{command_name}' has no goal_pos_w attribute and cannot be visualized."
            )
        if not hasattr(self.command.cfg, "object_name"):
            raise AttributeError(
                f"Command term '{command_name}' does not identify its target object."
            )

        self.target_object = env.scene[self.command.cfg.object_name]
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Sweep/ObjectGoal",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=marker_radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.15),
                        emissive_color=(0.0, 0.35, 0.03),
                        opacity=0.75,
                    ),
                )
            },
        )
        self.marker = VisualizationMarkers(marker_cfg)

    def update(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Move the marker and return goal, object position, and optional phase."""
        goal_pos_w = self.command.goal_pos_w
        object_pos_w = self.target_object.data.root_pos_w
        self.marker.visualize(translations=goal_pos_w)
        task_phase = getattr(self.command, "task_phase", None)
        return goal_pos_w, object_pos_w, task_phase


def _print_goal_status(
    step: int,
    goal_pos_w: torch.Tensor,
    object_pos_w: torch.Tensor,
    task_phase: torch.Tensor | None,
):
    """Print the goal status for environment zero."""
    goal = goal_pos_w[0].tolist()
    current = object_pos_w[0].tolist()
    error = torch.linalg.norm(goal_pos_w[0] - object_pos_w[0]).item()
    phase_text = ""
    if task_phase is not None:
        phase_text = f" | phase={'HOME' if task_phase[0].item() == 1 else 'SWEEP'}"
    print(
        f"[GOAL][step={step:06d}][env=0] "
        f"goal_w=({goal[0]:+.3f}, {goal[1]:+.3f}, {goal[2]:+.3f}) m | "
        f"object_w=({current[0]:+.3f}, {current[1]:+.3f}, {current[2]:+.3f}) m | "
        f"error={error:.3f} m{phase_text}",
        flush=True,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Load the checkpoint and run policy inference with goal visualization."""
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint_path = get_checkpoint_path(log_root, args_cli.load_run, "model_.*.pt")

    print(f"[INFO] Loading policy: {checkpoint_path}")
    print(
        f"[INFO] Object goal: green sphere, radius={args_cli.goal_radius:.3f} m, "
        f"command='{args_cli.command_name}'"
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    base_env = env.unwrapped
    visualizer = GoalVisualizer(base_env, args_cli.command_name, args_cli.goal_radius)
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
        with torch.inference_mode():
            goal_pos_w, object_pos_w, task_phase = visualizer.update()
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    runner.alg.actor_critic.reset(dones)
                goal_pos_w, object_pos_w, task_phase = visualizer.update()

            if args_cli.log_interval > 0 and step % args_cli.log_interval == 0:
                _print_goal_status(step, goal_pos_w, object_pos_w, task_phase)

            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

            sleep_time = base_env.step_dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
