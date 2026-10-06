#!/usr/bin/env python3
"""Trace model_4300 on actual V1 observations with its saved training geometry.

This is inference only. It restores the saved reset sampler and thumb-up
guard, disables newly introduced V1 height/outward guards, and captures the
physical transition before automatic reset. No production files are edited.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "logs/rsl_rl/hand_manipulation_push_v1/2026-10-05_22-17-17"
PACKAGE_ROOT = ROOT / "src/hand_manipulation_test"
sys.path.insert(0, str(PACKAGE_ROOT))
TASK_ID = "Isaac-Hand-Manipulation-Push-v1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, default=RUN)
parser.add_argument("--out", type=Path, default=OUT / "deterministic_trace")
parser.add_argument("--num_envs", "--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--checkpoint", type=Path, default=RUN / "model_4300.pt")
args = parser.parse_args()
if args.num_envs <= 0 or args.steps <= 0:
    parser.error("--num-envs and --steps must be positive")
launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import hand_manipulation_test  # noqa: E402, F401
from hand_manipulation_test import geometry  # noqa: E402
from hand_manipulation_test.mdp.push_events import PushSafePoseReset  # noqa: E402


class SavedConfigLoader(yaml.SafeLoader):
    pass


SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import saved source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_config(run):
    source = run / "source/hand_manipulation_test"
    saved_env = yaml.load((run / "params/env.yaml").read_text(), Loader=SavedConfigLoader)
    saved_agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=SavedConfigLoader)
    cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    for key, value in saved_env["task"].items():
        if not hasattr(cfg.task, key):
            raise ValueError(f"Saved task field is no longer supported: {key}")
        setattr(cfg.task, key, deepcopy(value))
    cfg.task.eef_soft_height_above_table_m = 100.0
    cfg.task.eef_hard_height_above_table_m = 101.0
    cfg.task.enforce_outward_fingers = False
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = saved_env["scene"]["env_spacing"]
    cfg.episode_length_s = saved_env["episode_length_s"]
    cfg.decimation = saved_env["decimation"]
    cfg.sim.dt = saved_env["sim"]["dt"]
    cfg.sim.render_interval = cfg.decimation
    cfg.seed = args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.__post_init__()  # Propagate restored task geometry into physical scene assets.

    # These files determine the old controller/observations/materials. A change
    # would invalidate this baseline, rather than merely alter its reward logs.
    unchanged = ("constants.py", "action_math.py", "geometry.py", "mdp/observations.py", "assets/robot.py")
    comparisons = []
    for relative in unchanged:
        frozen = source / relative
        active = PACKAGE_ROOT / "hand_manipulation_test" / relative
        same = sha256(frozen) == sha256(active)
        comparisons.append({"relative_path": relative, "same": same,
                            "saved_sha256": sha256(frozen), "active_sha256": sha256(active)})
        if not same:
            raise RuntimeError(f"Baseline-relevant active source differs from training: {relative}")
    # Load the complete original action implementations, not just their
    # numeric configs. Future fixes to production OSC must not change this
    # original-checkpoint baseline. Temporarily redirect the saved V1 class's
    # relative imports so its bases and target-integrator are also frozen.
    old_actions = source_module("hand_manipulation_test.mdp._trace_saved_actions", source / "mdp/actions.py")
    old_accumulation = source_module("hand_manipulation_test._trace_saved_accumulation_math", source / "accumulation_math.py")
    replacements = {"hand_manipulation_test.mdp.actions": old_actions,
                    "hand_manipulation_test.accumulation_math": old_accumulation}
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        old_v1_actions = source_module("hand_manipulation_test.mdp._trace_saved_push_v1_actions", source / "mdp/push_v1_actions.py")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    arm_values = {name: deepcopy(value) for name, value in saved_env["actions"]["arm_action"].items() if name != "class_type"}
    hand_values = {name: deepcopy(value) for name, value in saved_env["actions"]["hand_action"].items() if name != "class_type"}
    cfg.actions.arm_action = old_v1_actions.AccumulatedTranslationOscActionCfg(**arm_values)
    cfg.actions.hand_action = old_actions.InspireHandSynergyActionCfg(**hand_values)
    for name, term in vars(cfg.rewards).items():
        if hasattr(term, "weight"):
            term.weight = saved_env["rewards"].get(name, {}).get("weight", 0.0)
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.log_dir = str(args.out.expanduser().resolve())

    old_math = source_module("hand_manipulation_test._trace_saved_push_v1_math", source / "push_v1_math.py")
    old_events = source_module("hand_manipulation_test.mdp._trace_saved_push_v1_events", source / "mdp/push_v1_events.py")
    old_events.push_v1_palm_rotation = old_math.push_v1_palm_rotation
    # Setting the saved class before gym.make also protects the wrapper's
    # implicit initial reset from new wrist/outward reset restrictions.
    old_events.PushV1SafePoseReset._check_pose_and_limits = PushSafePoseReset._check_pose_and_limits
    cfg.events.safe_hand.func = old_events.PushV1SafePoseReset
    old_robot = source_module("hand_manipulation_test.assets._trace_saved_contact_robot", source / "assets/contact_robot.py")
    cfg.scene.robot = old_robot.make_contact_robot_cfg()
    cfg.task.validate()
    saved_agent["seed"] = args.seed
    saved_agent["device"] = cfg.sim.device
    return cfg, saved_agent, {
        "saved_env_path": str(run / "params/env.yaml"), "saved_agent_path": str(run / "params/agent.yaml"),
        "saved_reset_source": str(source / "mdp/push_v1_events.py"),
        "saved_rotation_source": str(source / "push_v1_math.py"),
        "saved_robot_source": str(source / "assets/contact_robot.py"),
        "saved_arm_action_source": str(source / "mdp/push_v1_actions.py"),
        "saved_parent_action_source": str(source / "mdp/actions.py"),
        "saved_target_integrator_source": str(source / "accumulation_math.py"),
        "reset_pose_guard": "PushSafePoseReset._check_pose_and_limits with saved H+X-up virtual guard",
        "baseline_relevant_source_comparisons": comparisons,
        "disabled_new_guards": {"eef_soft_height_above_table_m": 100., "eef_hard_height_above_table_m": 101.,
                                "enforce_outward_fingers": False},
        "task": cfg.task.to_dict(), "reward_weights": {n: t.weight for n, t in vars(cfg.rewards).items() if hasattr(t, "weight")},
        "scene_env_spacing_m": cfg.scene.env_spacing, "agent": deepcopy(saved_agent),
    }, old_events


class Trajectory:
    def __init__(self, env):
        self.env = env
        self.arm = env.action_manager.get_term("arm_action")
        self.safe = env.event_manager.get_term_cfg("safe_hand").func
        self.command = env.command_manager.get_term("target_position")
        self.robot = env.scene["robot"]
        wrist, _ = self.robot.find_joints("wrist_2_joint", preserve_order=True)
        self.wrist_id = wrist[0]
        self.episode_ids = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.rows = []
        self.fields = None
        self.initial = None
        self.pre = None
        self.step_index = -1
        self.original_compute = env.termination_manager.compute
        env.termination_manager.compute = self.capture_before_reset

    def desired_world(self):
        return self.robot.data.root_link_pos_w + math_utils.quat_apply(
            self.robot.data.root_link_quat_w, self.arm.desired_c_pose_b[:, :3])

    def set_inputs(self, observation, mean, action, step_index):
        c_position, c_quaternion = geometry.control_point_pose_w(self.env)
        root_quaternion = self.robot.data.root_link_quat_w
        c_quaternion_b = math_utils.quat_mul(math_utils.quat_conjugate(root_quaternion), c_quaternion)
        normalized = torch.nan_to_num(action).clamp(-1, 1)
        local_increment = normalized[:, :3] * action.new_tensor(self.arm.cfg.translation_scale)
        world_increment = math_utils.quat_apply(root_quaternion, math_utils.quat_apply(c_quaternion_b, local_increment))
        self.pre = {"observation": observation["policy"].clone(), "mean": mean.clone(), "action": action.clone(),
                    "normalized": normalized, "c_position": c_position.clone(), "c_quaternion": c_quaternion.clone(),
                    "desired": self.desired_world().clone(), "world_increment": world_increment,
                    "raw_world_up": math_utils.quat_apply(c_quaternion, mean[:, :3])[:, 2],
                    "normalized_world_up": math_utils.quat_apply(c_quaternion, normalized[:, :3])[:, 2]}
        self.step_index = step_index
        if self.initial is None:
            table = self.env.scene["table"].data.root_pos_w
            self.initial = {"eef_c_world_z_m": c_position[:, 2].cpu().tolist(),
                            "eef_c_above_table_m": (c_position[:, 2]-table[:, 2]-self.env.cfg.task.table_size[2]/2).cpu().tolist(),
                            "raw_actor_mean": mean.cpu().tolist(), "observations_64d": observation["policy"].cpu().tolist(),
                            "env_origins_w": self.env.scene.env_origins.cpu().tolist()}

    def capture_before_reset(self):
        done = self.original_compute()
        if self.pre is None:
            return done
        state = self.command.state()
        env, p = self.env, self.pre
        c_position, _ = geometry.control_point_pose_w(env)
        desired = self.desired_world()
        palm, hand_quaternion = self.safe.palm_reference_pose_w()
        cube = env.scene["target_object"].data
        table = env.scene["table"].data
        tabletop_z = table.root_pos_w[:, 2] + env.cfg.task.table_size[2]/2
        direction = self.command.direction_w
        cube_direction = math_utils.quat_apply_inverse(cube.root_quat_w, direction)
        extent = cube_direction.abs().sum(-1) * env.cfg.task.cube_size/2
        approach = cube.root_pos_w - extent[:, None]*direction
        approach[:, 2] += env.cfg.task.contact_palm_height_m
        base_pos, _ = geometry.base_link_pose_w(env)
        outward = self.command.target_pos_w-base_pos
        outward[:, 2] = 0
        outward = torch.nn.functional.normalize(outward, dim=-1, eps=1e-8)
        finger = math_utils.quat_apply(hand_quaternion, palm.new_tensor((0., 0., 1.)).expand_as(palm))
        wrist = self.robot.data.joint_pos[:, self.wrist_id]
        scalars = {
            "step": torch.full_like(wrist, self.step_index), "env_id": torch.arange(env.num_envs, device=env.device),
            "episode_id": self.episode_ids, "episode_step": env.episode_length_buf,
            "side_left": (self.command.angle_rad.cos() < 0), "command_angle_rad": self.command.angle_rad,
            "command_length_m": self.command.distance_m, "done": done,
            "time_out": env.termination_manager.time_outs, "success": state.success, "failure": state.failure,
            "table_failure": state.table_failure, "footprint_failure": state.footprint_failure,
            "fall_failure": state.fall_failure, "invalid_state": state.invalid_state,
            "eef_c_world_z_m": c_position[:, 2], "eef_c_local_z_m": c_position[:, 2]-env.scene.env_origins[:, 2],
            "tabletop_world_z_m": tabletop_z, "eef_c_above_table_m": c_position[:, 2]-tabletop_z,
            "desired_c_world_z_m": desired[:, 2], "desired_c_above_table_m": desired[:, 2]-tabletop_z,
            "controller_error_norm_m": (desired-c_position).norm(dim=-1),
            "eef_c_actual_dz_m": c_position[:, 2]-p["c_position"][:, 2],
            "desired_c_actual_dz_m": desired[:, 2]-p["desired"][:, 2],
            "commanded_world_dz_m": p["world_increment"][:, 2],
            "raw_mean_translation_world_up": p["raw_world_up"],
            "normalized_translation_world_up": p["normalized_world_up"],
            "raw_mean_clip_fraction": (p["mean"].abs() > 1).float().mean(-1),
            "executed_input_clip_fraction": (p["action"].abs() > 1).float().mean(-1),
            "palm_world_z_m": palm[:, 2], "palm_above_table_m": palm[:, 2]-tabletop_z,
            "palm_height_error_m": palm[:, 2]-cube.root_pos_w[:, 2]-env.cfg.task.contact_palm_height_m,
            "approach_gap_m": (palm-approach).norm(dim=-1), "raw_palm_contact": state.palm_cube_contact,
            "valid_push_contact": state.valid_push_contact, "grounded": state.grounded,
            "contact_seen": state.contact_seen, "push_seen": state.push_seen,
            "palm_normal_cos": state.palm_alignment_cos, "palm_force_cos": state.force_alignment_cos,
            "finger_outward_cos": (finger*outward).sum(-1),
            "wrist_2_raw_rad": wrist, "wrist_2_wrapped_rad": torch.atan2(wrist.sin(), wrist.cos()),
            "goal_error_m": state.cube_goal_distance_m,
            "cube_forward_m": ((cube.root_pos_w-self.command.target_pos_w)*direction).sum(-1),
        }
        for name, values in (("raw_mean", p["mean"]), ("policy_action", p["action"]),
                             ("normalized_action", p["normalized"]), ("obs", p["observation"]),
                             ("eef_c_pos_w", c_position), ("desired_c_pos_w", desired),
                             ("palm_pos_w", palm), ("cube_pos_w", cube.root_pos_w)):
            for index in range(values.shape[1]):
                scalars[f"{name}_{index:02d}"] = values[:, index]
        names = list(scalars)
        if self.fields is None:
            self.fields = names
        elif names != self.fields:
            raise RuntimeError("Trace schema changed within the rollout")
        packed = torch.stack([v.float() for v in scalars.values()], -1).detach().cpu().numpy()
        if not np.isfinite(packed).all():
            raise RuntimeError("Nonfinite physical/policy state in trace")
        self.rows.append(packed)
        return done

    def close(self):
        self.env.termination_manager.compute = self.original_compute


def summarize(data, fields):
    index = {name: i for i, name in enumerate(fields)}
    def stats(name):
        x = data[:, index[name]]
        return {"mean": float(x.mean()), "min": float(x.min()), "p05": float(np.quantile(x, .05)),
                "p95": float(np.quantile(x, .95)), "max": float(x.max())}
    names = ("eef_c_above_table_m", "desired_c_above_table_m", "palm_height_error_m", "approach_gap_m",
             "controller_error_norm_m", "commanded_world_dz_m", "raw_mean_translation_world_up",
             "normalized_translation_world_up", "raw_mean_clip_fraction", "executed_input_clip_fraction",
             "finger_outward_cos", "wrist_2_wrapped_rad", "goal_error_m", "cube_forward_m")
    result = {name: stats(name) for name in names}
    result["end_events"] = {name: int(data[:, index[name]].sum()) for name in (
        "done", "time_out", "success", "failure", "table_failure", "footprint_failure", "fall_failure", "invalid_state")}
    result["contact_policy_intervals"] = {name: int(data[:, index[name]].sum()) for name in (
        "raw_palm_contact", "valid_push_contact", "grounded")}
    result["per_axis_raw_mean"] = {f"action_{i}": stats(f"raw_mean_{i:02d}") for i in range(8)}
    return result


def main():
    run, out = args.run.expanduser().resolve(), args.out.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out.mkdir(parents=True, exist_ok=True)
    cfg, agent, provenance, old_events = restore_config(run)
    provenance.update({"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                       "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed, "mode": args.mode,
                       "capture_stage": "post_physics_pre_auto_reset", "inference_only": True})
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    env = gym.make(TASK_ID, cfg=cfg)
    trace = None
    try:
        bare = env.unwrapped
        safe = bare.event_manager.get_term_cfg("safe_hand").func
        safe._sample_pose = MethodType(old_events.PushV1SafePoseReset._sample_pose, safe)
        safe._finger_outward_cos = MethodType(old_events.PushV1SafePoseReset._finger_outward_cos, safe)
        safe._check_pose_and_limits = MethodType(PushSafePoseReset._check_pose_and_limits, safe)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
        runner = OnPolicyRunner(wrapped, deepcopy(agent), log_dir=None, device=bare.device)
        runner.load(str(checkpoint), map_location=bare.device)
        policy = runner.get_inference_policy(device=bare.device)
        observation = wrapped.get_observations()
        if observation["policy"].shape != (args.num_envs, 64):
            raise RuntimeError(f"Checkpoint requires actual 64D observations, got {observation['policy'].shape}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        std = saved["actor_state_dict"]["distribution.std_param"].tolist()
        trace = Trajectory(bare)
        started = time.monotonic()
        for step in range(args.steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                mean = policy(observation)
                action = mean if args.mode == "deterministic" else policy(observation, stochastic_output=True)
                trace.set_inputs(observation, mean, action, step)
                observation, _, dones, _ = wrapped.step(action)
                trace.episode_ids += dones.long()
                policy.reset(dones)
            if (step+1) % 100 == 0:
                z = trace.rows[-1][:, trace.fields.index("eef_c_above_table_m")]
                print(f"TRACE step={step+1} mode={args.mode} C_height_mean={z.mean():.4f}m max={z.max():.4f}m", flush=True)
        if not trace.rows:
            raise RuntimeError("No physics transitions were captured")
        data = np.concatenate(trace.rows)
        with (out / "trajectory.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(trace.fields)
            writer.writerows(data)
        result = {"provenance": provenance, "initial": trace.initial, "checkpoint_std_per_axis": std,
                  "completed_policy_steps": len(trace.rows), "elapsed_seconds": time.monotonic()-started,
                  "aggregate": summarize(data, trace.fields), "by_side": {}, "per_env": {}}
        side_index = trace.fields.index("side_left")
        env_index = trace.fields.index("env_id")
        for side, label in ((0, "right"), (1, "left")):
            selected = data[data[:, side_index] == side]
            if len(selected):
                result["by_side"][label] = summarize(selected, trace.fields)
        for env_id in range(args.num_envs):
            result["per_env"][str(env_id)] = summarize(data[data[:, env_index] == env_id], trace.fields)
        result["interpretation_limits"] = [
            "Deterministic raw-mean clipping is measured here and differs from analytic stochastic-std clipping bounds.",
            "The original training geometry/reset/control contract is restored; extra reward terms have zero weight and new physical guards are disabled.",
            "32 environments and a 500-step inference rollout do not reproduce the full 2048-environment stochastic training distribution.",
            "Rows capture physical states before auto-reset; obs_00..obs_63 are the pre-step inputs that produced the recorded raw means.",
        ]
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        print("TRACE_COMPLETE " + json.dumps({"out": str(out), "mode": args.mode,
            "steps": len(trace.rows), "mean_C_height_m": result["aggregate"]["eef_c_above_table_m"]["mean"],
            "max_C_height_m": result["aggregate"]["eef_c_above_table_m"]["max"],
            "raw_mean_clip_fraction": result["aggregate"]["raw_mean_clip_fraction"]["mean"],
            "end_events": result["aggregate"]["end_events"]}), flush=True)
    finally:
        if trace is not None:
            trace.close()
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)
