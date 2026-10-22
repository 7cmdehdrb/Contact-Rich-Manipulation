"""Supervise fresh simulator processes, rolling back interrupted PPO training.

Example: python train_resilient.py --headless --num_envs 4096 --max_iterations 90000
The target is absolute; each restored worker trains only the remaining range.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from training_recovery import (
    inspect_checkpoint, select_rollback, truncate_lineage, write_json_atomic,
)

DEFAULT_TASK = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0"


class StopRequested(Exception):
    def __init__(self, signum):
        self.signum = signum


def run_worker(command, log_path):
    """Tee the child output; explicit interruption never triggers a restart."""
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", start_new_session=True, bufsize=1)
    try:
        with Path(log_path).open("w") as logfile:
            for line in child.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()
        return child.wait()
    except BaseException as error:
        signum = (error.signum if isinstance(error, StopRequested)
                  else signal.SIGINT if isinstance(error, KeyboardInterrupt) else signal.SIGTERM)
        # A closed output pipe or full disk must not orphan the GPU worker.
        # Repeated Ctrl-C during teardown cannot interrupt child reaping.
        handlers = {item: signal.signal(item, signal.SIG_IGN) for item in (signal.SIGINT, signal.SIGTERM)}
        try:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
        finally:
            for item, handler in handlers.items():
                signal.signal(item, handler)
        raise
    finally:
        child.stdout.close()


def completion_valid(attempt_dir, target):
    try:
        value = json.loads((attempt_dir / "training_status.json").read_text())
        if value.get("status") != "completed" or value.get("target_iterations") != target:
            return False
        checkpoint = inspect_checkpoint(value["checkpoint"])
        return checkpoint.path.parent == attempt_dir and checkpoint.iteration >= target - 1
    except (OSError, ValueError, KeyError, TypeError):
        return False


def supervise(args, training_args, worker_script=None):
    """Run the actual coordinator; worker_script is dependency injection for tests."""
    if args.max_iterations <= 0 or args.rollback_iterations < 0 or args.max_restarts < 0:
        raise ValueError("Target must be positive; rollback and max-restarts must be nonnegative")
    if any(value == "--distributed" or value.startswith("--resilient-") for value in training_args):
        raise ValueError("Distributed workers and internal --resilient-* arguments are unsupported")
    # Resume belongs to the coordinator: never let train.py discover unrelated runs.
    forbidden = ("--checkpoint", "--load_run", "agent.resume=", "agent.load_run=", "agent.load_checkpoint=")
    if any(any(value == item or value.startswith(item + "=") or
                   (item.endswith("=") and value.startswith(item)) for item in forbidden)
           for value in training_args):
        raise ValueError("Use --resume-checkpoint PATH to select an explicit initial checkpoint")
    initial = inspect_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    job_dir = args.job_dir
    if job_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        job_dir = Path("logs/resilient") / args.task / f"{stamp}_{uuid.uuid4().hex[:8]}"
    job_dir = job_dir.resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    lineage = []
    if initial:
        lineage.append({"directory": str(initial.path.parent), "eligible_through": initial.iteration,
                        "source": "explicit_initial_checkpoint"})
    state = {"task": args.task, "target_iterations": args.max_iterations,
             "rollback_iterations": args.rollback_iterations, "max_restarts": args.max_restarts,
             "seed_policy": "Preserve forwarded/configured seed; no PPO parameter changes",
             "lineage": lineage, "attempts": [], "status": "running"}
    manifest = job_dir / "job.json"
    worker_script = Path(worker_script or Path(__file__).with_name("train.py")).resolve()
    resume = initial
    previous_handlers = {}

    def interrupt(signum, _frame):
        raise StopRequested(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        for number in range(args.max_restarts + 1):
            attempt_dir = job_dir / f"attempt_{number:03d}"
            attempt_dir.mkdir()
            lineage.append({"directory": str(attempt_dir), "eligible_through": None})
            command = [sys.executable, "-u", str(worker_script), *training_args,
                       "--task", args.task, "--max_iterations", str(args.max_iterations),
                       "--resilient-attempt-dir", str(attempt_dir),
                       "--resilient-target-iterations", str(args.max_iterations)]
            if resume:
                command.extend(["--resilient-checkpoint", str(resume.path)])
            attempt = {"number": number, "directory": str(attempt_dir), "command": command,
                       "restored_iteration": resume.iteration if resume else 0,
                       "resume_checkpoint": str(resume.path) if resume else None}
            state["attempts"].append(attempt)
            write_json_atomic(manifest, state)
            print(f"[recovery] Attempt {number}: iteration {attempt['restored_iteration']} -> "
                  f"absolute target {args.max_iterations}; {attempt_dir}", flush=True)
            returncode = run_worker(command, attempt_dir / "console.log")
            attempt["returncode"] = returncode
            if returncode in (-signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT, 128 + signal.SIGTERM):
                state["status"] = "stopped"
                write_json_atomic(manifest, state)
                return 128 + (signal.SIGINT if returncode in (-signal.SIGINT, 130) else signal.SIGTERM)
            if completion_valid(attempt_dir, args.max_iterations):
                state["status"] = "completed"
                if returncode != 0:
                    state["worker_exit_warning"] = (
                        f"Training target and final checkpoint verified, but worker exited with {returncode}"
                    )
                    print(f"[recovery] {state['worker_exit_warning']}; no rollback.", flush=True)
                write_json_atomic(manifest, state)
                print(f"[recovery] Completed target {args.max_iterations}; {manifest}", flush=True)
                return 0
            attempt["status"] = "incomplete"
            if number == args.max_restarts:
                state["status"] = "retry_limit_reached"
                write_json_atomic(manifest, state)
                print(f"[recovery] Stopped after {number} restarts; inspect {manifest}", file=sys.stderr)
                return 1
            resume, details = select_rollback(lineage, args.rollback_iterations)
            attempt["recovery"] = details
            truncate_lineage(lineage, resume.iteration if resume else 0)
            write_json_atomic(manifest, state)
            print(f"[recovery] Early exit ({returncode}); rollback latest {details['latest_iteration']} "
                  f"to {resume.iteration if resume else 'fresh iteration 0 (no valid rollback checkpoint)'}. "
                  "Old checkpoint suffixes remain on disk but are excluded from this lineage.", flush=True)
            for rejection in details["rejected"]:
                print(f"[recovery] Skipped invalid checkpoint: {rejection}", flush=True)
        return 1
    except (StopRequested, KeyboardInterrupt) as error:
        signum = error.signum if isinstance(error, StopRequested) else signal.SIGINT
        state["status"] = "stopped"
        state["signal"] = signum
        write_json_atomic(manifest, state)
        print("[recovery] Explicit stop received; no restart.", flush=True)
        return 128 + signum
    except Exception as error:
        state["status"] = "failed"
        state["error"] = str(error)
        write_json_atomic(manifest, state)
        print(f"[recovery] {error}; inspect {manifest}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max_iterations", type=int, default=90000,
                        help="Absolute total iteration target (default: v1.0's 90000)")
    parser.add_argument("--rollback-iterations", type=int, default=500)
    parser.add_argument("--max-restarts", type=int, default=10, help="Retries after initial attempt; 0 disables retry")
    parser.add_argument("--job-dir", type=Path, help="New directory; existing paths are never reused")
    parser.add_argument("--resume-checkpoint", "--resume", type=Path,
                        help="Explicit checkpoint for the first attempt; its directory is a read-only lineage source")
    args, training_args = parser.parse_known_args(argv)
    try:
        return supervise(args, training_args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
