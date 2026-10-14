#!/usr/bin/env python3
"""Read-only physics diagnostics; policy replay and scripted pose baselines.

Writes separate experiment artifacts. Does not change task configuration,
checkpoint weights, contact forces, robot state during rollout, or rewards.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/hand_manipulation_test"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("checkpoint_rollout.json"))
MODES = ("deterministic", "stochastic", "scripted_approach_target", "scripted_side_height_03")
parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
parser.add_argument("--side-penetration", type=float, default=.008)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat
from hand_manipulation_test.geometry import control_point_pose_w
from hand_manipulation_test.action_math import quaternion_rotate, quaternion_to_matrix
import hand_manipulation_test

def main():
    task = "Isaac-Hand-Manipulation-Push-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    agent.device = args.device
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
    env = gym.make(task, cfg=cfg)
    try:
        raw = env.unwrapped
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
        checkpoint = ROOT / "logs/rsl_rl/hand_manipulation_push/2026-10-05_13-31-47/model_1800.pt"
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=raw.device)
        command = raw.command_manager.get_term("target_position")
        safe = raw.event_manager.get_term_cfg("safe_hand").func
        arm = raw.action_manager.get_term("arm_action")
        hand = raw.action_manager.get_term("hand_action")
        robot = raw.scene["robot"]
        cube = raw.scene["target_object"]
        pad_id = robot.body_names.index("inspire_palm_force_sensor")
        pad_half = safe.collision_bounds.half_extents[pad_id]
        pad_center = safe.collision_bounds.centers[pad_id]
        original_compute = raw.reward_manager.compute
        rows = []
        action_input = torch.zeros((raw.num_envs, 8), device=raw.device)

        def capture(dt):
            reward = original_compute(dt)
            state = command.state()
            snapshot = command._snapshot()
            force = snapshot.cube_palm_forces_w_history[:, :, 0]
            norm = force.norm(dim=-1)
            active = norm >= cfg.task.contact_threshold_n
            force_dot = (force * command.direction_w[:, None, None]).sum(-1) / norm.clamp_min(1.e-8)
            pad_rotation = quaternion_to_matrix(robot.data.body_link_quat_w[:, pad_id])
            center = robot.data.body_link_pos_w[:, pad_id] + quaternion_rotate(robot.data.body_link_quat_w[:, pad_id], pad_center.expand(raw.num_envs, -1))
            vertical_extent = (pad_rotation[:, 2].abs() * pad_half).sum(-1)
            forward = ((cube.data.root_pos_w - command.target_pos_w) * command.direction_w).sum(-1)
            values = {
                "raw_action": action_input,
                "reward_terms": raw.reward_manager._step_reward * dt,
                "episode_step": raw.episode_length_buf,
                "right": command.angle_rad.cos() > 0,
                "forward_m": forward,
                "goal_error_m": state.cube_goal_distance_m,
                "palm_height_from_cube_m": snapshot.palm_pos_w[:, 2] - cube.data.root_pos_w[:, 2],
                "central_pad_bottom_from_cube_top_m": center[:, 2] - vertical_extent - cube.data.root_pos_w[:, 2] - cfg.task.cube_size / 2,
                "palm_alignment_cos": state.palm_alignment_cos,
                "force_alignment_cos": state.force_alignment_cos,
                "palm_force_n": norm.amax((1, 2)),
                "forward_force_n": (force * command.direction_w[:, None, None]).sum(-1).clamp_min(0).sum(-1).amax(-1),
                "raw_contact": active.any((1, 2)),
                "force_aligned": (active & (force_dot >= command.tracker.cfg.alignment_cos)).any((1, 2)),
                "normal_aligned": state.palm_alignment_cos >= command.tracker.cfg.alignment_cos,
                "valid_contact": state.valid_push_contact,
                "grounded": state.grounded,
                "contact_seen": state.contact_seen,
                "push_seen": state.push_seen,
                "success": state.success,
                "failure": state.failure,
                "table_failure": state.table_failure,
                "footprint_failure": state.footprint_failure,
                "fall_failure": state.fall_failure,
                "progress_paid": state.progress_delta > 0,
                "timeout": raw.termination_manager.time_outs,
            }
            rows.append({name: value.detach().cpu().numpy().copy() for name, value in values.items()})
            return reward

        raw.reward_manager.compute = capture
        results = {"checkpoint": str(checkpoint), "num_envs": raw.num_envs, "steps_per_mode": args.steps,
                   "dt": raw.step_dt, "translation_scale": cfg.actions.arm_action.translation_scale,
                   "scripted_side_penetration_m": args.side_penetration,
                   "approach_palm_height_m": command.tracker.cfg.palm_height_offset_m,
                   "modes": {}, "notes": ["Terminal transitions recorded before automatic reset.",
                       "Scripted baselines use the exact unmodified 8D actions, OSC, cube physics and rewards.",
                       "Scripted low target is a feasibility probe, not a proposed safe production controller."]}
        for mode in args.modes:
            started = time.perf_counter()
            rows.clear()
            raw.reset(seed=42)
            obs = wrapped.get_observations()
            for step in range(args.steps):
                with torch.no_grad():
                    if mode in ("deterministic", "stochastic"):
                        action_input = policy(obs, stochastic_output=(mode == "stochastic"))
                    else:
                        snapshot = command._snapshot()
                        quat = control_point_pose_w(raw)[1]
                        local_direction = quat_apply_inverse(cube.data.root_quat_w, command.direction_w)
                        surface_extent = local_direction.abs().sum(-1) * cfg.task.cube_size / 2
                        target_palm = cube.data.root_pos_w - surface_extent[:, None] * command.direction_w
                        target_palm[:, 2] += (command.tracker.cfg.palm_height_offset_m if mode == "scripted_approach_target" else .03)
                        delta_world = target_palm - snapshot.palm_pos_w
                        delta_world = delta_world.clamp(-.012, .012)
                        # Small penetration advances the object only after reaching the side.
                        if mode == "scripted_side_height_03":
                            near = (target_palm - snapshot.palm_pos_w).norm(dim=-1) < .02
                            delta_world += args.side_penetration * command.direction_w * near[:, None]
                        action_input = torch.zeros((raw.num_envs, 8), device=raw.device)
                        action_input[:, :3] = quat_apply_inverse(quat, delta_world) / delta_world.new_tensor(cfg.actions.arm_action.translation_scale)
                        initial_quat = safe.desired_c_quat_w
                        orientation_delta = axis_angle_from_quat(quat_mul(quat_conjugate(quat), initial_quat))
                        action_input[:, 3:6] = orientation_delta / delta_world.new_tensor(cfg.actions.arm_action.rotation_scale)
                        action_input[:, 6:] = hand.synergy_to_action(safe.initial_hand_synergy)
                    obs, reward, dones, extras = wrapped.step(action_input)
                    policy.reset(dones)
            data = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
            prefix = "rollout" if args.output.name == "checkpoint_rollout.json" else args.output.stem
            np.savez_compressed(args.output.with_name(f"{prefix}_{mode}.npz"), **data)
            summary = {"wall_time_s": time.perf_counter() - started}
            for side, selection in (("all", np.ones(data["right"].shape, dtype=bool)), ("right", data["right"]), ("left", ~data["right"])):
                done = (data["failure"] | data["success"] | data["timeout"]) & selection
                entry = {"transitions": int(selection.sum()), "completed_episodes": int(done.sum())}
                for name in ("raw_contact", "force_aligned", "normal_aligned", "valid_contact", "grounded", "progress_paid"):
                    entry[name + "_step_fraction"] = float(data[name][selection].mean())
                for name in ("contact_seen", "push_seen", "success", "failure", "table_failure", "footprint_failure", "fall_failure", "timeout"):
                    entry[name + "_completed_fraction"] = float(data[name][done].mean()) if done.any() else None
                for name in ("forward_m", "palm_force_n", "forward_force_n", "palm_height_from_cube_m", "central_pad_bottom_from_cube_top_m", "palm_alignment_cos"):
                    values = data[name][selection]
                    entry[name] = {"mean": float(values.mean()), "p01": float(np.quantile(values,.01)), "p50": float(np.quantile(values,.5)), "p99": float(np.quantile(values,.99)), "max": float(values.max())}
                entry["raw_action_clip_fraction"] = float((np.abs(data["raw_action"][selection]) > 1).mean())
                entry["reward_per_1000_transitions"] = dict(zip(raw.reward_manager.active_terms, (data["reward_terms"][selection].mean(0) * 1000).tolist()))
                summary[side] = entry
            results["modes"][mode] = summary
            args.output.write_text(json.dumps(results, indent=2))
            print("CHECKPOINT_DIAGNOSTIC", mode, json.dumps(summary["all"]), flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        omni.kit.app.get_app().post_quit(code)
        app.close()
    raise SystemExit(code)
