"""Play one Final-v1 checkpoint and split the merged goal/object rewards.

Run each policy at both curriculum stages with the same seed to separate policy
and stage effects. Results go under outputs/; the training run is read-only.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--curriculum-iteration", type=int)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--max-steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_v1_play_diagnostics"))
parser.add_argument("--stack-dump-seconds", type=int, default=0, help="Dump Python stacks periodically while diagnosing a stalled run")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint is None or args.curriculum_iteration is None:
    parser.error("--checkpoint and --curriculum-iteration are required")
if args.num_envs < 1 or args.episodes < 1 or args.max_steps < 1 or args.curriculum_iteration < 0 or args.stack_dump_seconds < 0:
    parser.error("num-envs, episodes and max-steps must be positive; curriculum-iteration must be nonnegative")
if not args.checkpoint.is_file():
    parser.error(f"checkpoint not found: {args.checkpoint}")

if args.stack_dump_seconds:
    import faulthandler
    import signal

    faulthandler.dump_traceback_later(args.stack_dump_seconds, repeat=True)

app = AppLauncher(args).app
try:
    import math
    import statistics

    import gymnasium as gym
    import torch

    import inspire_sweep_baseline  # noqa: F401
    from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
    from inspire_sweep_baseline.mdp.common.contact_sweep_clutter import all_object_tilt_angles
    from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_holding import target_measurements
    from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_object_state import active_object_mask
    from inspire_sweep_baseline.mdp.curriculum_functions.contact_sweep_clutter_fusion_holding import scheduled_value
    from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion import combined_active_object_cost
    from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion_holding_scheduled import goal_settling_centered
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    task = "Isaac-Inspire-Final-v1"
    device = args.device or "cuda:0"
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args.seed
    agent_cfg.device = device
    agent_cfg.seed = args.seed

    def goal_schedule(name, iteration):
        cfg = getattr(env_cfg.curriculum, name)
        return scheduled_value(iteration, cfg.params["points"], cfg.params["interval"])

    references = {
        3000: (goal_schedule("goal_position_tolerance", 3000), goal_schedule("goal_speed_scale", 3000)),
        9500: (goal_schedule("goal_position_tolerance", 9500), goal_schedule("goal_speed_scale", 9500)),
    }

    class RewardTap:
        """Observe the original callable at reward-computation time, before reset."""

        def __init__(self, original, kind, weight):
            self.original = original
            self.kind = kind
            self.weight = weight
            self.latest = {}

        def reset(self, env_ids=None):
            if hasattr(self.original, "reset"):
                self.original.reset(env_ids=env_ids)

        def __call__(self, env, **params):
            value = self.original(env, **params)
            dt = env.step_dt
            if self.kind == "object":
                if self.weight != -1.0:
                    raise ValueError("Object component split expects manager weight -1")
                combined = combined_active_object_cost(
                    env,
                    free_angle=params["free_angle"],
                    fail_angle=params["fail_angle"],
                    displacement_ratio=params["displacement_ratio"],
                    displacement_max_cost=params["displacement_max_cost"],
                )
                angles = all_object_tilt_angles(env)
                active = active_object_mask(env, angles)
                tilt_each = ((angles - params["free_angle"]) / (
                    params["fail_angle"] - params["free_angle"]
                )).clamp(0.0, 1.0).square()
                target_ids = env.target_id.reshape(env.num_envs, 1)
                object_ids = torch.arange(angles.shape[1], device=angles.device).view(1, -1)
                target_tilt = torch.where(active & (object_ids == target_ids), tilt_each, 0.0).sum(dim=-1)
                obstacle_tilt = torch.where(active & (object_ids != target_ids), tilt_each, 0.0).sum(dim=-1)
                tilt = target_tilt + obstacle_tilt
                displacement = combined - tilt
                acceleration = value - 30.0 * combined
                observed = {
                    "object_target_tilt": -30.0 * target_tilt * dt,
                    "object_obstacle_tilt": -30.0 * obstacle_tilt * dt,
                    "object_tilt": -30.0 * tilt * dt,
                    "object_displacement": -30.0 * displacement * dt,
                    "object_acceleration": -acceleration * dt,
                    "active_obstacles": active.sum(dim=-1).float() - 1.0,
                }
                parts = observed["object_tilt"] + observed["object_displacement"] + observed["object_acceleration"]
                torch.testing.assert_close(parts, self.weight * value * dt, rtol=1e-4, atol=1e-4)
            else:
                distance, speed, valid = target_measurements(env, params.get("command_name", "target_goal_pos"))
                radius = params["position_tolerance"]
                within_2radius = valid & (distance <= 2.0 * radius)
                distance_factor = torch.exp(-math.log(2) * (distance / radius).square())
                speed_factor = torch.exp(-math.log(2) * (speed / params["speed_scale"]).square())
                center_factor = 1.0 + (1.0 - distance / radius).clamp(0.0, 1.0).square()
                observed = {
                    "goal_hold_observed": self.weight * value * dt,
                    "goal_distance_m": torch.where(valid, distance, 0.0),
                    "goal_speed_m_s": torch.where(valid, speed, 0.0),
                    "goal_valid": valid.float(),
                    "goal_inside_radius": (valid & (distance <= radius)).float(),
                    "goal_inside_2radius": within_2radius.float(),
                    "goal_distance_factor": torch.where(valid, distance_factor, 0.0),
                    "goal_speed_factor": torch.where(valid, speed_factor, 0.0),
                    "goal_center_factor": torch.where(valid, center_factor, 0.0),
                }
                reconstructed = self.weight * dt * torch.where(
                    within_2radius, distance_factor * speed_factor * center_factor, 0.0)
                torch.testing.assert_close(observed["goal_hold_observed"], reconstructed, rtol=1e-4, atol=1e-5)
                for stage, (test_radius, test_speed) in references.items():
                    observed[f"goal_hold_at_{stage}_formula"] = self.weight * dt * goal_settling_centered(
                        env, position_tolerance=test_radius, speed_scale=test_speed,
                        command_name=params.get("command_name", "target_goal_pos"),
                    )
                torch.testing.assert_close(observed["goal_hold_observed"], self.weight * value * dt)
            self.latest = {name: tensor.detach().clone() for name, tensor in observed.items()}
            return value

    print(f"Creating {task} on {device} for checkpoint {args.checkpoint.name}", flush=True)
    env = gym.make(task, cfg=env_cfg)
    print("Environment created; attaching reward observers", flush=True)
    if args.stack_dump_seconds:
        faulthandler.register(signal.SIGUSR1, all_threads=False)
    raw = env.unwrapped
    taps = {}
    for name, kind in (("object_penalty", "object"), ("goal_hold", "goal")):
        cfg = raw.reward_manager.get_term_cfg(name)
        tap = RewardTap(cfg.func, kind, cfg.weight)
        cfg.func = tap
        raw.reward_manager.set_term_cfg(name, cfg)
        taps[name] = tap

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args.stack_dump_seconds:
        def trace_runner(frame, event, _arg):
            filename = frame.f_code.co_filename
            if event in ("call", "return") and (
                "rsl_rl/algorithms/ppo.py" in filename
                or "rsl_rl/models/mlp_model.py" in filename
                or filename.endswith("/agent/final_models.py")
                or filename.endswith("/agent/contact_sweep_clutter_fusion_models.py")
            ):
                print(f"RUNNER TRACE {event} {filename}:{frame.f_lineno} {frame.f_code.co_name}", flush=True)

        sys.setprofile(trace_runner)
    runner = CurriculumOnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    if args.stack_dump_seconds:
        sys.setprofile(None)
    runner.load(str(args.checkpoint.resolve()))
    print(f"Checkpoint loaded; applying curriculum iteration {args.curriculum_iteration}", flush=True)
    checkpoint_iteration = runner.current_learning_iteration
    raw.set_learning_iteration(args.curriculum_iteration)
    stage_params = {
        "goal_radius_m": raw.reward_manager.get_term_cfg("goal_hold").params["position_tolerance"],
        "goal_speed_scale_m_s": raw.reward_manager.get_term_cfg("goal_hold").params["speed_scale"],
        "object_free_acceleration_m_s2": raw.reward_manager.get_term_cfg("object_penalty").params["free_acceleration"],
        "object_acceleration_weight": raw.reward_manager.get_term_cfg("object_penalty").params["acceleration_weight"],
        "requested_gray_obstacles": raw.event_manager.get_term_cfg("object_spawn").params["gray_obstacles"],
        "requested_white_obstacles": raw.event_manager.get_term_cfg("object_spawn").params["white_obstacles"],
    }
    policy = runner.get_inference_policy(device=raw.device)
    env.reset(seed=args.seed)
    print("Reset complete; starting inference", flush=True)
    obs = wrapped.get_observations()
    term_names = list(raw.reward_manager.active_terms)
    if not {"goal_hold", "object_penalty"}.issubset(term_names):
        raise RuntimeError(f"Missing expected reward terms: {term_names}")
    n = raw.num_envs
    accum = {}
    lengths = torch.zeros(n, dtype=torch.int64, device=raw.device)
    rows = []
    for step in range(args.max_steps):
        with torch.inference_mode():
            action = policy(obs)
            obs, reward, done, _ = wrapped.step(action)
            policy.reset(done)
            term_steps = raw.reward_manager._step_reward.detach() * raw.step_dt
            torch.testing.assert_close(term_steps.sum(dim=-1), reward, rtol=1e-4, atol=1e-4)
            samples = {f"term_{name}": term_steps[:, i].clone() for i, name in enumerate(term_names)}
            for tap in taps.values():
                if not tap.latest:
                    raise RuntimeError("Reward observer did not run")
                samples.update(tap.latest)
            samples["total_reward"] = reward.detach().clone()
            for name, value in samples.items():
                if name not in accum:
                    accum[name] = torch.zeros(n, device=raw.device)
                accum[name] += value
            lengths += 1
            for env_id in torch.nonzero(done, as_tuple=False).flatten().tolist():
                count = int(lengths[env_id].item())
                row = {"checkpoint_iteration": checkpoint_iteration, "curriculum_iteration": args.curriculum_iteration,
                       "seed": args.seed, "env_id": env_id, "length_steps": count}
                row.update({name: float(value[env_id].item()) for name, value in accum.items()})
                row["goal_valid_fraction"] = row["goal_valid"] / count
                row["goal_inside_fraction"] = row["goal_inside_radius"] / count
                row["goal_inside_2radius_fraction"] = row["goal_inside_2radius"] / count
                row["mean_active_obstacles"] = row["active_obstacles"] / count
                if row["goal_valid"] > 0:
                    row["mean_valid_goal_distance_m"] = row["goal_distance_m"] / row["goal_valid"]
                    row["mean_valid_goal_speed_m_s"] = row["goal_speed_m_s"] / row["goal_valid"]
                    row["mean_valid_goal_distance_factor"] = row["goal_distance_factor"] / row["goal_valid"]
                    row["mean_valid_goal_speed_factor"] = row["goal_speed_factor"] / row["goal_valid"]
                else:
                    row["mean_valid_goal_distance_m"] = None
                    row["mean_valid_goal_speed_m_s"] = None
                    row["mean_valid_goal_distance_factor"] = None
                    row["mean_valid_goal_speed_factor"] = None
                for term_name in raw.termination_manager.active_terms:
                    row[f"termination_{term_name}"] = bool(raw.termination_manager.get_term(term_name)[env_id].item())
                part_sum = row["object_tilt"] + row["object_displacement"] + row["object_acceleration"]
                if abs(row["object_target_tilt"] + row["object_obstacle_tilt"] - row["object_tilt"]) > 1e-3:
                    raise AssertionError(f"Object tilt split did not reconcile: {row}")
                if abs(part_sum - row["term_object_penalty"]) > 1e-3:
                    raise AssertionError(f"Object reward split did not reconcile: {row}")
                if abs(row["goal_hold_observed"] - row["term_goal_hold"]) > 1e-3:
                    raise AssertionError(f"Goal hold tap did not reconcile: {row}")
                if abs(sum(row[f"term_{name}"] for name in term_names) - row["total_reward"]) > 1e-3:
                    raise AssertionError(f"Reward terms did not reconcile: {row}")
                rows.append(row)
                for value in accum.values():
                    value[env_id] = 0.0
                lengths[env_id] = 0
            if len(rows) >= args.episodes:
                break
        if (step + 1) % 100 == 0:
            print(f"{step + 1} steps, {len(rows)} complete episodes", flush=True)
    wrapped.close()
    if len(rows) < args.episodes:
        raise RuntimeError(f"Only {len(rows)} complete episodes after {args.max_steps} steps")
    rows = rows[:args.episodes]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"policy_{checkpoint_iteration}_stage_{args.curriculum_iteration}_seed_{args.seed}"
    csv_path = args.output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_fields = {}
    for name in rows[0]:
        if name in {"checkpoint_iteration", "curriculum_iteration", "seed", "env_id"}:
            continue
        values = [row[name] for row in rows if row[name] is not None]
        mean_fields[name] = statistics.mean(values) if values else None
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iteration": checkpoint_iteration,
        "curriculum_iteration": args.curriculum_iteration,
        "seed": args.seed,
        "episodes": len(rows),
        "num_envs": args.num_envs,
        "stage_params": stage_params,
        "mean_per_episode": mean_fields,
        "terms": term_names,
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path),
                      "checkpoint_iteration": checkpoint_iteration,
                      "curriculum_iteration": args.curriculum_iteration,
                      "mean_total_reward": mean_fields["total_reward"],
                      "mean_goal_hold": mean_fields["term_goal_hold"],
                      "mean_object_penalty": mean_fields["term_object_penalty"]}, indent=2))
finally:
    if args.stack_dump_seconds:
        faulthandler.unregister(signal.SIGUSR1)
        faulthandler.cancel_dump_traceback_later()
    app.close()
