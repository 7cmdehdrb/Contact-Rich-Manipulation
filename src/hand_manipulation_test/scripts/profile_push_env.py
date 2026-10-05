#!/usr/bin/env python3
"""Measure Push reset and policy-step costs without smoke assertion overhead."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--reset-repeats", type=int, default=3)
parser.add_argument("--legacy-reset", action="store_true", help="Compare the original PhysX finite-difference IK.")
parser.add_argument("--enable-markers", action="store_true")
parser.add_argument("--stagger-timeouts", action="store_true", help="Include steady-state auto-reset costs with uniformly staggered episode lengths.")
parser.add_argument("--output", type=Path)
parser.add_argument("--cprofile", action="store_true", help="Also save Python call costs for the measured rollout.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
simulation_app = launcher.app


def main():
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import hand_manipulation_test  # noqa: F401
    from hand_manipulation_test.mdp.contact_events import ContactSafePoseReset

    if args.num_envs <= 0 or not 1 <= args.steps < 400 or args.reset_repeats < 1:
        raise ValueError("num_envs/reset-repeats must be positive; steps must be in [1,399]")
    cfg = load_cfg_from_registry("Isaac-Hand-Manipulation-Push-v0", "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.commands.target_position.debug_vis = args.enable_markers
    cfg.scene.ee_frame.debug_vis = args.enable_markers
    started = time.perf_counter()
    env = gym.make("Isaac-Hand-Manipulation-Push-v0", cfg=cfg).unwrapped
    try:
        safe = env.event_manager.get_term_cfg("safe_hand").func
        if args.legacy_reset:
            if hasattr(safe, "use_cached_kinematics"):
                safe.use_cached_kinematics = False
            else:
                safe._solve_seed = ContactSafePoseReset._solve_seed.__get__(safe)
        sync = lambda: torch.cuda.synchronize(env.device) if str(env.device).startswith("cuda") else None
        sync()
        creation_s = time.perf_counter() - started
        env.reset(seed=42)
        sync()
        reset_times = {}
        for count in sorted({1, min(8, env.num_envs), env.num_envs}):
            ids = torch.arange(count, device=env.device)
            timings = []
            for index in range(args.reset_repeats):
                sync()
                started = time.perf_counter()
                env._reset_idx(ids)
                sync()
                timings.append(time.perf_counter() - started)
            reset_times[str(count)] = timings
        # Warm the same normal stepping path measured below.
        action = torch.zeros((env.num_envs, 8), device=env.device)
        hand = env.action_manager.get_term("hand_action")
        convert = getattr(hand, "synergy_to_action", lambda value: 2 * value - 1)
        for _ in range(10):
            action[:, 6:] = convert(hand.actual_synergy)
            env.step(action)
        if args.stagger_timeouts:
            env.episode_length_buf[:] = torch.randint(env.max_episode_length, (env.num_envs,), device=env.device)
        sync()
        baseline_counter = safe.ik_iterations.clone()
        baseline_io = dict(getattr(safe, "reset_io_counters", {}))
        terminations = torch.zeros((), device=env.device, dtype=torch.long)
        reset_calls = 0
        reset_rows = 0
        original_reset = env._reset_idx
        def count_reset(ids):
            nonlocal reset_calls, reset_rows
            reset_calls += 1
            reset_rows += len(ids)
            return original_reset(ids)
        env._reset_idx = count_reset
        if args.cprofile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        started = time.perf_counter()
        for _ in range(args.steps):
            action[:, 6:] = convert(hand.actual_synergy)
            _, _, terminated, truncated, _ = env.step(action)
            terminations += (terminated | truncated).sum()
        sync()
        step_s = time.perf_counter() - started
        if args.cprofile:
            import pstats
            profiler.disable()
            destination = (args.output or Path("/tmp/hand_push_profile.json")).with_suffix(".pstats")
            destination.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(destination))
            pstats.Stats(profiler).sort_stats("cumulative").print_stats(35)
        result = {
            "num_envs": env.num_envs, "steps": args.steps, "device": str(env.device),
            "legacy_reset": args.legacy_reset, "markers": args.enable_markers,
            "stagger_timeouts": args.stagger_timeouts,
            "creation_s": creation_s, "reset_seconds_by_batch_size": reset_times,
            "policy_step_mean_s": step_s / args.steps,
            "environment_steps_per_second": env.num_envs * args.steps / step_s,
            "rollout_36_steps_s": step_s / args.steps * 36,
            "terminated_rows_during_steps": int(terminations),
            "rollout_reset_calls": reset_calls, "rollout_reset_rows": reset_rows,
            "reset_work_during_steps": bool((safe.ik_iterations != baseline_counter).any()),
            "reset_io_counters": getattr(safe, "reset_io_counters", {}),
            "rollout_reset_io_counters": {
                key: value - baseline_io[key]
                for key, value in getattr(safe, "reset_io_counters", {}).items()
            },
        }
        print("PUSH_PROFILE " + json.dumps(result), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
    finally:
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
