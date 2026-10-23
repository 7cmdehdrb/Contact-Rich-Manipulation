#!/usr/bin/env python3
"""Replay Fusion v1_0/v1_3_1 and collect reset-safe physical reward inputs per environment.

Defaults: requested model_14500.pt, 16 environments, 25000 steps (500 sim seconds
per environment). Training/checkpoint directories are only read.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASK = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3_1"
DEFAULT_CHECKPOINT = "logs/resilient/Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_3_1/2026-10-22_10-27-21_5c4cf77b/attempt_000/model_14500.pt"


def collect(args, checkpoint, output, report):
    import torch

    report["torch_version"] = torch.__version__
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; Isaac Sim playback measurements were not collected")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless, device=args.device).app
    report["_app"] = app
    env = None
    writer = None
    try:
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        import inspire_sweep_baseline  # noqa: F401
        from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
        from inspire_sweep_baseline.runtime.contact_sweep_clutter_fusion_telemetry import RewardCSVWriter, RewardTelemetryEnv

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        rsl_version = importlib.metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)
        cfg.seed = agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        cfg.log_dir = str(output)
        dump_yaml(str(output / "env.yaml"), cfg)
        dump_yaml(str(output / "agent.yaml"), agent_cfg)
        # Same registry config/runner as play.py. Only the runtime gains the
        # minimal pre-reset capture hook; the parent step loop is reused.
        physical_env = RewardTelemetryEnv(cfg)
        env = RslRlVecEnvWrapper(physical_env, clip_actions=agent_cfg.clip_actions)
        runner = CurriculumOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        telemetry = physical_env.playback_telemetry
        report.update(
            rsl_rl_version=rsl_version, gpu=torch.cuda.get_device_name(),
            runtime_class=type(physical_env).__name__,
            checkpoint_iteration=runner.current_learning_iteration,
            physics_dt_s=physical_env.physics_dt, control_dt_s=physical_env.step_dt,
            object_names=telemetry.object_names, object_labels=telemetry.object_labels,
            wrench_units=["N", "N", "N", "N*m", "N*m", "N*m"],
            wrench="existing policy PureWristWrench: downstream Hand gravity/inertia compensated",
            wrench_origin="penalty/raw_eef/world columns: EEF point; raw_joint columns: measurement child joint point",
            wrench_sign="parent-on-child support reaction; opposite sign to external load",
            velocity="object_lin_vel_w: world-axis center-of-mass linear velocity in m/s; no rounding",
            time="elapsed simulation seconds, starting at control_dt_s; not wall time",
            sampling="every control step, last physics substep; terminal captured before auto-reset",
            delta_valid="env_NN.csv: 0 for first sample after reset; exclude from F/T differences",
            curriculum="fixed at checkpoint iteration during playback",
        )
        writer = RewardCSVWriter(output, telemetry)
        report["columns"] = writer.columns
        report["termination_names"] = telemetry.termination_names
        report["object_ids"] = {str(i): name for i, name in enumerate(telemetry.object_names)}
        report["object_active_note"] = "Inactive objects are identified by active_objN=0; exclude them from penalty analysis."
        report["force_note"] = "Each object: norm(-sum(force_matrix_w over hand filters)); aggregate obstacle_net_force_N: norm(sum of active non-target vectors), not sum of norms. Normal contact only, no tangential friction."
        report["acceleration_note"] = "norm(object_com_lin_acc_w), PhysX last-substep COM acceleration; acceleration_valid_objN suppresses first post-reset/activation sample."
        report["raw_wrench_note"] = "raw_joint: uncorrected incoming joint reaction at inspire_base_link child joint origin/axes; raw_eef: same uncorrected reaction transported to EEF. penalty_eef: compensated EEF input to wrench_change."
        report["eef_speed_note"] = "norm(body_link_lin_vel + cross(body_link_ang_vel, EEF_position-body_link_position)), same as eef_linear_speed_cost."
        report["effective_reward_settings"] = {
            name: {"weight": physical_env.reward_manager.get_term_cfg(name).weight,
                   "params": str(physical_env.reward_manager.get_term_cfg(name).params)}
            for name in physical_env.reward_manager.active_terms
        }
        # Restore checkpoint curricula before the first measured episode.
        env.reset()
        observations = env.get_observations()
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    raise RuntimeError(f"Simulator stopped at {step}/{args.steps} steps")
                started = time.perf_counter()
                observations, _, dones, _ = env.step(policy(observations))
                writer.write(telemetry, (step + 1) * physical_env.step_dt)
                policy.reset(dones)
                if physical_env.learning_iteration != report["checkpoint_iteration"]:
                    raise RuntimeError("Curriculum advanced during playback")
                report.update(control_steps_completed=step + 1, csv_rows_per_file=writer.rows)
                if (step + 1) % 100 == 0:
                    print(f"[Telemetry] {step + 1}/{args.steps} steps; simulated={(step + 1) * physical_env.step_dt:.2f}s per env", flush=True)
                if args.real_time:
                    time.sleep(max(0.0, physical_env.step_dt - (time.perf_counter() - started)))
        if writer.rows != args.steps:
            raise RuntimeError("Incomplete CSV acquisition")
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=[TASK, "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0"], default=TASK)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=25000, help="Control steps per environment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--real-time", dest="real_time", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num_envs and --steps must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    output = (args.output_dir or Path("outputs/fusion_reward_inputs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if "logs" in output.parts or output == checkpoint.parent or checkpoint.parent in output.parents:
        parser.error("Choose an output directory outside logs and the checkpoint run")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running", task=args.task, checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args) | {"checkpoint": str(checkpoint), "output_dir": str(output)},
        command=sys.argv, control_steps_completed=0, csv_rows_per_file=0,
    )
    for name in ("env.yaml", "agent.yaml"):
        source = checkpoint.parent / "params" / name
        if source.is_file():
            shutil.copyfile(source, output / f"source_{name}")
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    error = None
    try:
        collect(args, checkpoint, output, report)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed during collection")
        report["status"] = "complete"
    except BaseException:
        error = traceback.format_exc()
        report.update(status="failed", error=error)
        print(error, file=sys.stderr, flush=True)
    app = report.pop("_app", None)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[Telemetry] {report['status']}: {output} ({report['csv_rows_per_file']} rows/file)", flush=True)
    if error:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Kit fast shutdown must not hide an acquisition failure.
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
