"""CPU-only analysis of saved Push logs; never imports or modifies the task.

Run with the Isaac Lab Python environment. Optional --run/--out paths make
the same analysis reproducible for another run with the same logging schema.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import yaml


class SavedConfigLoader(yaml.SafeLoader):
    """Accept only the tuple tag used by saved configuration values."""


SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "p05": float(np.quantile(a, .05)), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "last": float(a[-1])}


def checkpoint_distribution(path):
    # The checkpoint is the user's local training artifact. No simulator/GPU is used.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    if "distribution.std_param" in actor:
        std = actor["distribution.std_param"].double().numpy()
        parameterization = "scalar"
    else:
        std = actor["distribution.log_std_param"].double().exp().numpy()
        parameterization = "log"
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid saved Gaussian standard deviation")
    names = ["C_dx", "C_dy", "C_dz", "C_rx", "C_ry", "C_rz", "hand_common_open", "hand_thumb_open"]
    lower = np.array([math.erfc(1 / (s * math.sqrt(2))) for s in std])
    # Among Normal(mu, sigma), mu=0 maximizes P(-1 <= A <= 1).
    # These clipping probabilities therefore lower-bound any observation-conditioned mean.
    per_axis = [{"action": name, "std": float(s), "clip_probability_lower_bound": float(p),
                 "zero_mean_boundary_mass_each_sign": float(p / 2),
                 "max_sensitivity_d_expected_clipped_action_d_mean": float(1 - p)}
                for name, s, p in zip(names, std, lower)]
    return {"path": str(path), "saved_iteration": checkpoint["iter"],
            "std_parameterization": parameterization, "std_mean": float(std.mean()),
            "latent_gaussian_entropy_nats": float(np.log(std * math.sqrt(2 * math.pi * math.e)).sum()),
            "per_axis": per_axis,
            "mean_axis_clip_probability_lower_bound": float(lower.mean()),
            "expected_clipped_axes_lower_bound": float(lower.sum()),
            "any_axis_clipped_probability_lower_bound": float(1 - np.prod(1 - lower)),
            "all_eight_axes_clipped_probability_lower_bound": float(np.prod(lower)),
            "note": "Analytic bounds under saved diagonal Gaussian, not measured rollout histograms. "
                    "Per-axis means/observations were not saved. Bounds concern stochastic sampling; "
                    "deterministic play uses the network mean. Torque clipping is a separate quantity."}


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "logs/rsl_rl/hand_manipulation_push/2026-10-05_13-31-47")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = yaml.load((run / "params/env.yaml").read_text(), Loader=SavedConfigLoader)
    agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=SavedConfigLoader)
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    series = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
    iteration_series = {tag: rows for tag, rows in series.items() if not tag.endswith("/time")}
    iterations = sorted({row.step for rows in iteration_series.values() for row in rows})
    lo, hi = iterations[0], iterations[-1]
    count = len(iterations)
    tags = sorted(iteration_series)
    lookup = {tag: {row.step: row.value for row in rows} for tag, rows in iteration_series.items()}
    with (out / "scalars.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["iteration", *tags])
        for step in iterations:
            writer.writerow([step, *(lookup[tag].get(step, "") for tag in tags)])
    with (out / "scalars_long.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "step", "wall_time_unix_s", "value"])
        for tag, rows in series.items():
            writer.writerows((tag, row.step, row.wall_time, row.value) for row in rows)

    thirds = np.array_split(np.asarray(iterations), 3)
    windows = {"all": (lo, hi), "first_100": (lo, min(lo + 99, hi)),
               "early_third": (int(thirds[0][0]), int(thirds[0][-1])),
               "middle_third": (int(thirds[1][0]), int(thirds[1][-1])),
               "late_third": (int(thirds[2][0]), int(thirds[2][-1])),
               "last_300": (max(lo, hi - 299), hi), "last_100": (max(lo, hi - 99), hi)}
    checkpoints = sorted(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if checkpoints:
        checkpoint_it = int(checkpoints[-1].stem.split("_")[-1])
        windows["checkpoint_last_100"] = (max(lo, checkpoint_it - 99), min(hi, checkpoint_it))
    episode_s = float(env["episode_length_s"])
    policy_dt = float(env["sim"]["dt"] * env["decimation"])
    transitions_per_iteration = int(env["scene"]["num_envs"] * agent["num_steps_per_env"])
    weights = {term: float(config["weight"]) for term, config in env["rewards"].items()
               if isinstance(config, dict) and "weight" in config}
    summary = {"run": str(run), "first_iteration": lo, "last_iteration": hi, "iterations": count,
               "num_envs": env["scene"]["num_envs"], "rollout_steps": agent["num_steps_per_env"],
               "policy_dt_s": policy_dt, "reward_log_denominator_s": episode_s,
               "logged_transitions": count * transitions_per_iteration,
               "logging_semantics": [
                   "Episode_Reward values already contain term weight and manager dt integration; multiply by max episode seconds (10) to recover episode contribution.",
                   "Reward window means are means of per-iteration means of resetting-row means. Reset-batch sizes are unavailable, so these are not episode-count-weighted population estimates.",
                   "Contact/push/success command metrics are selected resetting rows' previously recorded state; no force norm, gap, alignment cosine, or full trajectories are logged.",
                   "Episode_Termination values average persistent most-recent-episode statuses across all environments, whereas command/reward metrics refer to resetting rows. Their averages need not match.",
                   "Train/mean_reward and episode length use the rolling completed-episode deque; their sampling differs from Episode_Reward averages.",
                   "Loss/entropy is differential entropy of raw Gaussian actions; it is not entropy of the clipped physical action distribution."],
               "saved_agent": agent, "saved_task": env["task"], "reward_weights": weights,
               "saved_arm_stiffness": env["actions"]["arm_action"]["motion_stiffness"],
               "saved_hand_synergy_range": env["actions"]["hand_action"]["synergy_range"],
               "windows": {}, "all_run_ranges": {}}
    for name, (start, end) in windows.items():
        statistics = {tag: stats([row.value for row in rows if start <= row.step <= end])
                      for tag, rows in iteration_series.items() if any(start <= row.step <= end for row in rows)}
        contributions = {tag.removeprefix("Episode_Reward/"): values["mean"] * episode_s
                         for tag, values in statistics.items() if tag.startswith("Episode_Reward/")}
        positive = sum(max(value, 0) for value in contributions.values())
        negative = sum(abs(min(value, 0)) for value in contributions.values())
        collection = statistics["Perf/collection_time"]["mean"]
        learning = statistics["Perf/learning_time"]["mean"]
        contact_duration = contributions.get("contact", 0) / weights["contact"]
        summary["windows"][name] = {"start": start, "end": end, "statistics": statistics,
            "means": {tag: data["mean"] for tag, data in statistics.items()},
            "episode_reward_contributions": contributions,
            "positive_share": {key: max(value, 0) / positive if positive else 0 for key, value in contributions.items()},
            "negative_share": {key: abs(min(value, 0)) / negative if negative else 0 for key, value in contributions.items()},
            "sum_episode_contributions": sum(contributions.values()),
            "implied_maintaining_contact_seconds_per_episode_mean": contact_duration,
            "implied_credited_progress_potential_per_episode_mean": contributions.get("progress", 0) / weights["progress"],
            "mean_completed_episode_length_s": statistics["Train/mean_episode_length"]["mean"] * policy_dt,
            "performance": {"collection_seconds_per_iteration": collection, "learning_seconds_per_iteration": learning,
                            "learning_fraction": learning / (learning + collection),
                            "transition_throughput_per_second": transitions_per_iteration / (learning + collection)}}
    for tag, rows in iteration_series.items():
        values = np.asarray([row.value for row in rows])
        nonzero = np.flatnonzero(values)
        summary["all_run_ranges"][tag] = {**stats(values), "nonzero_iterations": int(len(nonzero)),
            "first_nonzero_iteration": int(rows[nonzero[0]].step) if len(nonzero) else None,
            "last_nonzero_iteration": int(rows[nonzero[-1]].step) if len(nonzero) else None}
    perf_rows = iteration_series["Perf/collection_time"]
    collection_sum = sum(row.value for row in perf_rows)
    learning_sum = sum(row.value for row in iteration_series["Perf/learning_time"])
    summary["performance_total"] = {"collection_seconds": collection_sum, "learning_seconds": learning_sum,
        "logged_iteration_seconds": collection_sum + learning_sum,
        "collection_fraction": collection_sum / (collection_sum + learning_sum),
        "transition_throughput_per_second": summary["logged_transitions"] / (collection_sum + learning_sum),
        "first_scalar_wall_time_utc": datetime.fromtimestamp(perf_rows[0].wall_time, timezone.utc).isoformat(),
        "last_scalar_wall_time_utc": datetime.fromtimestamp(perf_rows[-1].wall_time, timezone.utc).isoformat(),
        "wall_time_span_seconds": perf_rows[-1].wall_time - perf_rows[0].wall_time}
    summary["discount_reference"] = {"gamma": agent["algorithm"]["gamma"],
        "effective_horizon_seconds_1_over_1_minus_gamma": policy_dt / (1 - agent["algorithm"]["gamma"]),
        "reward_discount_at_1_second": agent["algorithm"]["gamma"] ** (1 / policy_dt),
        "reward_discount_at_2_seconds": agent["algorithm"]["gamma"] ** (2 / policy_dt),
        "note": "Credit-assignment characteristic of requested reference PPO; this analysis does not change PPO."}
    source_comparison = []
    for path in sorted((run / "source/hand_manipulation_test").rglob("*.py")):
        relative = path.relative_to(run / "source/hand_manipulation_test")
        active = root / "src/hand_manipulation_test/hand_manipulation_test" / relative
        saved_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        active_hash = hashlib.sha256(active.read_bytes()).hexdigest() if active.exists() else None
        source_comparison.append({"relative_path": str(relative), "saved_sha256": saved_hash,
                                  "active_sha256": active_hash, "same": saved_hash == active_hash})
    summary["source_comparison"] = source_comparison
    summary["changed_active_source_paths"] = [item["relative_path"] for item in source_comparison if not item["same"]]
    distributions = [checkpoint_distribution(path) for path in checkpoints
                     if int(path.stem.split("_")[-1]) in {0, 100, 600, 1200, 1800, checkpoint_it}]
    (out / "checkpoint_action_distribution.json").write_text(json.dumps(distributions, indent=2))
    summary["checkpoint_distributions"] = distributions
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "reward_windows.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", "start", "end", "term", "episode_contribution", "positive_share", "negative_share"])
        for name, data in summary["windows"].items():
            for term, value in data["episode_reward_contributions"].items():
                writer.writerow([name, data["start"], data["end"], term, value, data["positive_share"][term], data["negative_share"][term]])
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hand_push_analysis_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)
    def line(axis, tag, scale=1, width=50, **kwargs):
        rows = iteration_series[tag]
        x = np.asarray([r.step for r in rows])
        y = np.asarray([r.value * scale for r in rows])
        axis.plot(x[width-1:], np.convolve(y, np.ones(width) / width, mode="valid"), **kwargs)
    for term in weights:
        tag = "Episode_Reward/" + term
        if tag in iteration_series:
            line(axes[0], tag, episode_s, label=term)
    axes[0].set(title="Weighted episode contributions (50-iteration means)", ylabel="Reward contribution")
    axes[0].legend(ncol=5)
    for metric in ("contact_seen", "push_seen", "success"):
        line(axes[1], "Metrics/target_position/" + metric, 100, label=metric)
    axes[1].set(ylabel="Logged resetting-row fraction (%)")
    distance_axis = axes[1].twinx()
    line(distance_axis, "Metrics/target_position/goal_distance_m", 100, label="goal distance", color="grey", linestyle="--")
    distance_axis.set_ylabel("Goal distance (cm)")
    axes[1].legend()
    line(axes[2], "Policy/mean_std", width=1, label="raw Gaussian mean std")
    axes[2].axhline(1, color="grey", linestyle="--", label="action clamp magnitude")
    axes[2].set(ylabel="Standard deviation")
    axes[2].legend()
    for metric in ("collection_time", "learning_time"):
        line(axes[3], "Perf/" + metric, label=metric)
    axes[3].set(xlabel="PPO iteration", ylabel="Seconds / iteration")
    axes[3].legend()
    fig.savefig(out / "training_diagnosis.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"iterations": count, "last_iteration": hi, "logged_transitions": summary["logged_transitions"],
                      "last_100": summary["windows"]["last_100"], "last_300": summary["windows"]["last_300"],
                      "latest_checkpoint": distributions[-1], "performance_total": summary["performance_total"],
                      "changed_active_source_paths": summary["changed_active_source_paths"]}, indent=2))


if __name__ == "__main__":
    main()
