"""Checkpoint and progress contracts shared by the isolated training supervisor.

No Isaac/Kit imports: checkpoint inspection runs on CPU after a worker exits.
The caller supplies the job's lineage explicitly; historical runs are never
searched. Superseded checkpoint suffixes remain on disk but become ineligible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    iteration: int


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inspect_checkpoint(path):
    """Validate a trusted local PPO checkpoint without loading it onto a GPU."""
    import torch

    path = Path(path).resolve()
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Unreadable checkpoint {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    iteration = data.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"Checkpoint has no nonnegative integer iter: {path}")
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match and int(match.group(1)) != iteration:
        raise ValueError(f"Checkpoint filename/iter disagree: {path}")
    if not (isinstance(data.get("actor_state_dict"), dict) and data["actor_state_dict"]
            and isinstance(data.get("critic_state_dict"), dict) and data["critic_state_dict"]):
        if not isinstance(data.get("model_state_dict"), dict) or not data["model_state_dict"]:
            raise ValueError(f"Checkpoint has no PPO model state: {path}")
    if not isinstance(data.get("optimizer_state_dict"), dict):
        raise ValueError(f"Checkpoint has no optimizer state: {path}")

    tensor_count = 0

    def visit(value, name):
        nonlocal tensor_count
        if isinstance(value, torch.Tensor):
            tensor_count += 1
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite checkpoint tensor {name}: {path}")
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("std", "std_param") and (value <= 0).any():
                raise ValueError(f"Nonpositive Gaussian standard deviation {name}: {path}")
            if leaf in ("log_std", "log_std_param"):
                std = value.exp()
                if not torch.isfinite(std).all() or (std <= 0).any():
                    raise ValueError(f"Invalid exponentiated Gaussian log standard deviation {name}: {path}")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{name}.{index}")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite checkpoint value {name}: {path}")

    for key, value in data.items():
        if key != "infos":
            visit(value, key)
    if tensor_count == 0:
        raise ValueError(f"Checkpoint contains no model tensors: {path}")
    return Checkpoint(path, iteration)


def checkpoint_candidates(lineage):
    """Return only eligible model_N.pt files in explicitly owned run branches."""
    candidates = []
    for branch_index, branch in enumerate(lineage):
        directory = Path(branch["directory"])
        cap = branch.get("eligible_through")
        for path in directory.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match:
                iteration = int(match.group(1))
                if cap is None or iteration <= cap:
                    candidates.append((iteration, branch_index, path))
    return sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)


def select_rollback(lineage, rollback_iterations=500):
    """Select greatest saved iteration <= latest-rollback, with zero fallback.

    Latest means the eligible file index, including a potentially corrupt last
    save. Validate the chosen checkpoint and skip bad earlier candidates. This
    avoids accidentally moving the rollback target forward after corruption.
    """
    if rollback_iterations < 0:
        raise ValueError("rollback_iterations must be nonnegative")
    candidates = checkpoint_candidates(lineage)
    if not candidates:
        return None, {"latest_iteration": None, "rollback_target": 0, "rejected": []}
    latest = candidates[0][0]
    target = max(0, latest - rollback_iterations)
    details = {"latest_iteration": latest, "rollback_target": target, "rejected": []}
    for iteration, _, path in candidates:
        if iteration > target:
            continue
        try:
            selected = inspect_checkpoint(path)
        except ValueError as error:
            details["rejected"].append(str(error))
            continue
        return selected, details
    if target == 0:
        # RSL-RL overwrites model_0 after its first update. A failed first save
        # must not turn a recoverable bootstrap failure into a permanent stop.
        details["fallback"] = "fresh_iteration_zero_no_valid_checkpoint"
        return None, details
    raise ValueError(f"No valid checkpoint at or before rollback target {target}; {details}")


def truncate_lineage(lineage, iteration):
    """Exclude old future branches without deleting or modifying any checkpoint."""
    for branch in lineage:
        cap = branch.get("eligible_through")
        branch["eligible_through"] = iteration if cap is None else min(iteration, cap)


def remaining_iterations(target_iterations, restored_iteration):
    """RSL-RL starts its next loop at checkpoint.iter, not checkpoint.iter+1."""
    if target_iterations <= 0 or restored_iteration < 0:
        raise ValueError("Expected positive target and nonnegative restored iteration")
    return max(0, target_iterations - restored_iteration)


def restore_for_training(runner, checkpoint=None):
    """Restore iteration, then resample observations/goals at its curriculum.

    Runner construction needs initial observations. Reset again AFTER load so
    those provisional iteration-zero goals never enter a resumed rollout.
    """
    if checkpoint is not None:
        runner.load(str(checkpoint))
    environment = runner.env.unwrapped
    if hasattr(environment, "set_learning_iteration"):
        environment.set_learning_iteration(runner.current_learning_iteration)
    runner.env.reset()
    return runner.current_learning_iteration


def save_bootstrap_checkpoint(runner, path):
    """Save the framework's PPO state before learn initializes its logger.

    OnPolicyRunner.save uploads through Logger.save_model, which requires the
    writer created by learn(). Initializing that writer here would create it
    twice. Use the public algorithm save API and the runner's iter/infos schema
    for this local bootstrap only; ordinary/final saves keep runner.save.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = runner.alg.save()
    data["iter"] = runner.current_learning_iteration
    data["infos"] = None
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)
