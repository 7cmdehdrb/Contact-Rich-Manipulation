#!/usr/bin/env python3
"""Run a local PPO checkpoint on the right-only Inspire shelf-sweeping task."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
from pathlib import Path
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sweep_inspire_rl import TASK_ID, TASK_IDS, TASK_V1_ID, TASK_V2_ID, TASK_V3_ID
OBJECT_NAMES = ("bottle_1", "cup_1", "cup_2", "mug_1", "mug_2", "can_1", "weighted_cylinder")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=TASK_IDS, default=TASK_ID)
parser.add_argument("--object-name", choices=OBJECT_NAMES, default=None, help="Defaults to cup_1; V3 uses weighted_cylinder.")
parser.add_argument("--num_envs", "--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=0, help="Zero runs until the simulator closes.")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--disable-markers", action="store_true")
parser.add_argument("--gate-log", type=Path, help="Write pre-reset sweep gate diagnostics to CSV each step.")
parser.add_argument("--gate-log-env", type=int, default=0, help="Vector environment index to diagnose (default: 0).")
AppLauncher.add_app_launcher_args(parser)
# Add required task arguments after AppLauncher's preliminary parse so --help
# can show all launcher options without requiring a checkpoint.
parser.add_argument("--checkpoint", type=Path, required=True)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

from sweep_inspire_rl.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    InspireShelfSweepPPORunnerCfg, InspireShelfSweepV1PPORunnerCfg,
    InspireShelfSweepV2PPORunnerCfg, InspireShelfSweepV3PPORunnerCfg,
)
from sweep_inspire_rl.env_cfg import (  # noqa: E402
    InspireShelfSweepEnvCfg_PLAY, InspireShelfSweepV1EnvCfg_PLAY,
    InspireShelfSweepV2EnvCfg_PLAY, InspireShelfSweepV3EnvCfg_PLAY,
)


def main() -> None:
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.steps < 0:
        raise ValueError("--steps cannot be negative")
    if not 0 <= args.gate_log_env < args.num_envs:
        raise ValueError("--gate-log-env must be a valid environment index")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg_type, agent_cfg_type = {
        TASK_ID: (InspireShelfSweepEnvCfg_PLAY, InspireShelfSweepPPORunnerCfg),
        TASK_V1_ID: (InspireShelfSweepV1EnvCfg_PLAY, InspireShelfSweepV1PPORunnerCfg),
        TASK_V2_ID: (InspireShelfSweepV2EnvCfg_PLAY, InspireShelfSweepV2PPORunnerCfg),
        TASK_V3_ID: (InspireShelfSweepV3EnvCfg_PLAY, InspireShelfSweepV3PPORunnerCfg),
    }[args.task]
    env_cfg = env_cfg_type(**({"object_name": args.object_name} if args.object_name is not None else {}))
    agent_cfg = agent_cfg_type()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.target_goal_pos.debug_vis = not args.disable_markers
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed
    env_cfg.log_dir = str(checkpoint.parent)

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    env = gym.make(args.task, cfg=env_cfg)
    gate_log = None
    try:
        if args.gate_log is not None:
            path = args.gate_log.expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            gate_log = path.open("w", newline="", encoding="utf-8")
            env.unwrapped.sweep_gate_diagnostic_env = args.gate_log_env
            print(f"[INFO] Sweep gate diagnostics: {path} (env {args.gate_log_env})", flush=True)
        gate_writer = None
        gate_hits = 0
        blocked_counts = dict.fromkeys(("near_hand", "near_wrist"), 0)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        compatible_checkpoint = handle_deprecated_rsl_rl_checkpoint(str(checkpoint), installed_version)
        print(f"[INFO] Loading model checkpoint: {compatible_checkpoint}")
        runner.load(compatible_checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        observation = wrapped_env.get_observations()
        steps = 0
        while simulation_app.is_running() and (args.steps == 0 or steps < args.steps):
            started = time.time()
            with torch.inference_mode():
                action = policy(observation)
                observation, _, dones, _ = wrapped_env.step(action)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
                elif hasattr(runner.alg, "policy"):
                    runner.alg.policy.reset(dones)
            steps += 1
            if gate_log is not None:
                snapshot = env.unwrapped.sweep_gate_diagnostics
                row = {"step": steps, "env_id": args.gate_log_env}
                row.update({name: value.item() for name, value in snapshot.items()})
                if gate_writer is None:
                    gate_writer = csv.DictWriter(gate_log, fieldnames=list(row))
                    gate_writer.writeheader()
                gate_writer.writerow(row)
                gate_hits += int(row["gate"])
                for name in blocked_counts:
                    blocked_counts[name] += int(not row[name])
                if steps % 50 == 0:
                    gate_log.flush()
                    print(
                        f"[GATE] step={steps} active={row['gate']} "
                        f"reach={row['reaching_distance_m']:.4f}m "
                        f"wrist_y={row['wrist_y_distance_m']:.4f}m goal_region={row['goal_region']} "
                        f"wrist_raw={row['wrist_y_distance_uncompensated_m']:.4f}m "
                        f"palm={row['palm_contact']} other_pad={row['other_pad_contact']} "
                        f"progress={row['object_progress_m']:.3f}m "
                        f"forward_v={row['object_forward_velocity_m_s']:.3f}m/s "
                        f"tilt={row['object_tilt_deg']:.1f}deg "
                        f"sweep={row['sweeping_raw']:.3f}", flush=True,
                    )
            sleep_time = env.unwrapped.step_dt - (time.time() - started)
            if args.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)
        print(f"[INFO] Completed {steps} policy steps", flush=True)
        if gate_log is not None:
            print(f"[GATE] enabled={gate_hits}/{steps}; blocked counts (can overlap): {blocked_counts}", flush=True)
    finally:
        if gate_log is not None:
            gate_log.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
