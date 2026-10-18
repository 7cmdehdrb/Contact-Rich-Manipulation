"""Manual GPU probe of bounded OSC, including pre-reset/substep velocities.

Run with Isaac Lab Python; this is a short diagnostic, not a policy evaluation.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.assets.robot import ARM_JOINT_NAMES

task = "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2"
cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
cfg.seed = args.seed
env = gym.make(task, cfg=cfg).unwrapped
robot = env.scene["robot"]
joint_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
arm = env.action_manager.get_term("arm_action")
samples, control_samples, termination_samples = [], [], []
original_apply, original_compute = arm.apply_actions, env.termination_manager.compute


def read_velocity():
    return robot.data.joint_vel[:, joint_ids].abs().amax(dim=-1).clone()


def apply_and_sample():
    samples.append(read_velocity())
    original_apply()


def compute_and_sample():
    velocity = read_velocity()
    samples.append(velocity)
    control_samples.append(velocity)
    result = original_compute()
    termination_samples.append({
        name: env.termination_manager.get_term(name).clone()
        for name in ("hand_velocity", "push_fast", "shelf_collision")
    })
    return result


arm.apply_actions = apply_and_sample
env.termination_manager.compute = compute_and_sample
try:
    with torch.inference_mode():
        for mode in ("zero", "corners", "random"):
            env.reset(seed=args.seed)
            samples.clear()
            control_samples.clear()
            termination_samples.clear()
            torch.manual_seed(args.seed)
            actions = torch.ones((env.num_envs, 8), device=env.device)
            bits = torch.arange(env.num_envs, device=env.device)[:, None]
            axes = torch.arange(6, device=env.device)[None, :]
            corners = 2.0 * ((bits >> axes) & 1).float() - 1.0
            arm._compute_ee_jacobian()
            initial_jacobian = arm._jacobian_b.clone()
            for _ in range(args.steps):
                if mode == "zero":
                    actions[:, :6] = 0.0
                elif mode == "corners":
                    actions[:, :6] = corners * 10.0  # Saturate both norm limits.
                else:
                    actions[:, :6] = torch.randn((env.num_envs, 6), device=env.device)
                env.step(actions)
            velocity = torch.stack(samples)
            controls = torch.stack(control_samples)
            startup = velocity[: min(15, len(velocity))]
            # For Kp=100 and damping=20, quasi-static pose_rel tracking gives
            # desired Cartesian velocity approx (Kp/Kd)*delta = 5*delta.
            position_limit = arm.cfg.position_command_limit
            rotation_limit = arm.cfg.orientation_command_limit
            cartesian = torch.cat((corners[:, :3] * position_limit, corners[:, 3:] * rotation_limit), dim=-1)
            for section, limit in ((slice(0, 3), position_limit), (slice(3, 6), rotation_limit)):
                delta = cartesian[:, section]
                delta *= limit / delta.norm(dim=-1, keepdim=True).clamp_min(limit)
            estimate = (torch.linalg.pinv(initial_jacobian) @ (5.0 * cartesian).unsqueeze(-1)).squeeze(-1)
            def first_excess(threshold):
                indices = (controls > threshold).any(dim=-1).nonzero().flatten()
                return int(indices[0]) + 1 if len(indices) else None
            report = {
                "mode": mode, "num_envs": env.num_envs, "steps": args.steps,
                "max_joint_speed": velocity.max().item(),
                "startup_5_steps_max": startup.max().item(),
                "p95_joint_speed": torch.quantile(velocity.flatten(), 0.95).item(),
                "fraction_above_1_5": (velocity > 1.5).float().mean().item(),
                "fraction_above_5": (velocity > 5.0).float().mean().item(),
                "control_steps": len(controls),
                "first_control_step_above_1_5": first_excess(1.5),
                "first_control_step_above_5": first_excess(5.0),
                "jacobian_velocity_estimate_max_at_reset": estimate.abs().max().item(),
                "processed_position_norm_max": arm.processed_actions[:, :3].norm(dim=-1).max().item(),
                "processed_rotation_norm_max": arm.processed_actions[:, 3:].norm(dim=-1).max().item(),
                "termination_counts": {
                    name: sum(row[name].sum().item() for row in termination_samples)
                    for name in ("hand_velocity", "push_fast", "shelf_collision")
                },
            }
            print("OSC_PROBE " + json.dumps(report), flush=True)
finally:
    arm.apply_actions = original_apply
    env.termination_manager.compute = original_compute
    env.close()
    app.app.close()
