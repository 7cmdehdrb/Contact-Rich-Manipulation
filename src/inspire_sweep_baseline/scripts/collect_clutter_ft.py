#!/usr/bin/env python3
"""Replay a Curriculum-v1 checkpoint and stream paired v1/v2 F/T to CSV.

All outputs go to a new directory; checkpoint/log directories are read-only.
No policy export or new training is performed. The policy keeps v1 raw F/T.
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
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TASK = "Isaac-Inspire-Contact-Sweep-Clutter-FT-Test-v0"
DEFAULT_RUN = "logs/inspire_contact_sweep_clutter_v1_curriculum/2026-10-20_22-49-30"


def checkpoint_path(value):
    """Select the greatest numeric iteration when a run directory is given."""
    path = Path(value).resolve()
    if path.is_dir():
        candidates = [p for p in path.glob("model_*.pt") if p.stem[6:].isdigit()]
        if not candidates:
            raise ValueError(f"No model_<iteration>.pt checkpoints in {path}")
        path = max(candidates, key=lambda p: int(p.stem[6:]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def collect(args, checkpoint, output, report):
    import torch

    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable: real Isaac Sim F/T acquisition requires a working NVIDIA GPU/driver. "
            "No physical samples were collected."
        )

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=args.headless, device=args.device)
    app = launcher.app
    env = None
    writer = None
    try:
        import gymnasium as gym
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        import inspire_sweep_baseline  # noqa: F401
        from inspire_sweep_baseline.agents.curriculum_runner import CurriculumOnPolicyRunner
        from inspire_sweep_baseline.contact_sweep_clutter_ft_test.telemetry import WrenchCSVWriter

        cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        rsl_version = importlib.metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)
        cfg.seed = agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        cfg.log_dir = str(output)
        dump_yaml(str(output / "env.yaml"), cfg)
        dump_yaml(str(output / "agent.yaml"), agent_cfg)
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=agent_cfg.clip_actions)
        runner = CurriculumOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        physical_env = env.unwrapped
        report.update(
            rsl_rl_version=rsl_version,
            checkpoint_iteration=runner.current_learning_iteration,
            physics_dt_s=physical_env.physics_dt,
            control_dt_s=physical_env.step_dt,
            observation_groups=agent_cfg.obs_groups,
            sample_phase="post-physics control transition, before auto-reset for terminal rows",
            wrench_axes_order=["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            wrench_units=["N", "N", "N", "N*m", "N*m", "N*m"],
            wrench_origin="EEF for every wrench and moment; world denotes axes, not origin",
            quaternion_order="w,x,y,z",
            sign="parent-on-child support reaction; external load acts with opposite sign",
            v1_raw="exact legacy policy F/T transform, retained unchanged",
            v2_pure="v2 raw minus modeled downstream Hand gravity/inertia support",
            deltas="world-axis change per control step, NOT time derivative; exclude delta_valid=0",
            policy_observation="v1 raw F/T; no extra F/T channels or noise calls in the policy",
            rewards="original Curriculum-v1; no new wrench penalty",
            sampling="last physics substep of every control step; not every physics substep",
        )
        writer = WrenchCSVWriter(output / "wrench.csv")
        observations = env.get_observations()
        completed = 0
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    raise RuntimeError(f"Simulator stopped after {completed}/{args.steps} control steps")
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
                writer.write(physical_env.ft_telemetry)
                policy.reset(dones)
                completed = step + 1
                report.update(control_steps_completed=completed, csv_rows=writer.rows)
                if completed % 100 == 0:
                    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(f"[F/T] {completed}/{args.steps} steps, {writer.rows} rows", flush=True)
        if writer.rows != args.steps * args.num_envs:
            raise RuntimeError("Incomplete CSV row count")
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        # Main writes success/failure metadata before closing Kit, whose fast
        # shutdown can otherwise mask exceptions with a zero process exit.
        report["_app"] = app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_RUN, help="Checkpoint .pt or run directory (latest numeric model).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Must not already exist.")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000, help="Control steps per environment, not learning iterations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num_envs and --steps must be positive")
    checkpoint = checkpoint_path(args.checkpoint)
    output = args.output_dir or Path("outputs/clutter_ft") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    if output == checkpoint.parent or checkpoint.parent in output.parents:
        parser.error("--output-dir must be outside the training run directory")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running", task=TASK, checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args) | {"output_dir": str(output)},
        control_steps_completed=0, csv_rows=0,
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
        report["status"] = "complete"
    except BaseException:
        error = traceback.format_exc()
        report.update(status="failed", error=error)
        print(error, file=sys.stderr, flush=True)
    app = report.pop("_app", None)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[F/T] {report['status']}: {output} ({report['csv_rows']} rows)", flush=True)
    if error:
        # Avoid Kit's fast shutdown hiding failed acquisition.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
