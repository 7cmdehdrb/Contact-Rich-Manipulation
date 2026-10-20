#!/usr/bin/env python3
"""Compare measured TensorBoard rewards with the Clutter-v1 calibration.

Run with any Python that has tensorboard installed; Isaac Sim is not imported.
Episode_Reward values already include weights and are divided by the configured
maximum episode length, not the actual episode duration. Iterations are equally
weighted here, so the result is not an equally weighted mean over all episodes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


V1_EPISODE_SECONDS = 10.0
V1_WEIGHTS = {
    "reaching": 30.0,
    "goal_distance": 100.0,
    "goal_hold": 0.9,
    "contact_any": 0.16,
    "contact_count": 0.16,
    "obstacle_displacement": -5.0,
}
V1_NEW_POSITIVE_TERMS = ("goal_hold", "contact_any", "contact_count")
OBSTACLE_MAX_SQUARED_DISTANCE = 0.04


def read_source_config(log_dir: Path, term_names: set[str]) -> dict:
    """Read numeric fields from IsaacLab's generated YAML without loading objects."""
    path = log_dir / "params" / "env.yaml"
    if not path.is_file():
        is_v1 = all(name in term_names for name in V1_NEW_POSITIVE_TERMS)
        return {
            "path": None,
            "assumption": f"No params/env.yaml; canonical Clutter-v{int(is_v1)} weights and 10 seconds assumed.",
            "episode_length_s": 10.0,
            "reward_weights": {
                "reaching": 30.0,
                "goal_distance": 100.0 if is_v1 else 94.0,
            },
        }
    # Read only the plain scalar fields emitted by IsaacLab's config dumper.
    # In particular, never instantiate its !!python/object or callable tags.
    episode_length = None
    weights = {}
    in_rewards = False
    current_term = None
    for line in path.read_text().splitlines():
        if line.startswith("episode_length_s:"):
            episode_length = float(line.partition(":")[2].split("#", 1)[0])
        if line == "rewards:":
            in_rewards = True
            continue
        if in_rewards and line and not line[0].isspace() and not line.startswith("#"):
            in_rewards = False
        if not in_rewards:
            continue
        term = re.fullmatch(r"  ([A-Za-z_][A-Za-z_0-9]*):\s*", line)
        if term:
            current_term = term.group(1)
        weight = re.fullmatch(r"    weight:\s*([^#]+?)(?:\s*#.*)?", line)
        if weight and current_term is not None:
            weights[current_term] = float(weight.group(1))
    if episode_length is None or not math.isfinite(episode_length) or episode_length <= 0:
        raise ValueError(f"{path}: missing or invalid plain episode_length_s scalar")
    for name in ("reaching", "goal_distance"):
        if name not in weights or not math.isfinite(weights[name]) or weights[name] <= 0:
            raise ValueError(f"{path}: missing or invalid positive rewards.{name}.weight")
    return {
        "path": str(path.resolve()),
        "assumption": None,
        "episode_length_s": episode_length,
        "reward_weights": weights,
    }


def percentages(values: dict[str, float], denominator: float) -> dict:
    return {
        name: (100.0 * value / denominator if denominator else None)
        for name, value in values.items()
    }


def positive_groups(means: dict[str, float]) -> dict:
    groups = {
        "reaching": means["reaching"],
        "goal_distance": means["goal_distance"],
        "goal_hold": means["goal_hold"],
        "contact": means["contact_any"] + means["contact_count"],
    }
    total = math.fsum(groups.values())
    return {
        "weighted_logged_means": groups,
        "positive_group_sum": total,
        "share_percent_of_positive_groups": percentages(groups, total),
        "contact_components": {
            name: {
                "weighted_logged_mean": means[name],
                "share_percent_of_positive_groups": (
                    100.0 * means[name] / total if total else None
                ),
            }
            for name in ("contact_any", "contact_count")
        },
    }


def analyze(log_dir: Path, last_n: int) -> dict:
    if not log_dir.is_dir():
        raise ValueError(f"Log directory does not exist: {log_dir}")
    if last_n < 0:
        raise ValueError("--last-n must be nonnegative (0 means all records)")
    # Retain all records, including resumed-step overlaps, so duplicate steps
    # cannot silently bias the iteration mean.
    accumulator = EventAccumulator(
        str(log_dir), size_guidance={"scalars": 0}, purge_orphaned_data=False
    ).Reload()
    tags = sorted(
        tag for tag in accumulator.Tags()["scalars"] if tag.startswith("Episode_Reward/")
    )
    if not tags:
        raise ValueError(f"No Episode_Reward/* scalar tags found in {log_dir}")
    series = {}
    reference_steps = None
    for tag in tags:
        events = accumulator.Scalars(tag)
        if not events:
            raise ValueError(f"Empty scalar series: {tag}")
        steps = [event.step for event in events]
        if len(set(steps)) != len(steps):
            raise ValueError(f"Duplicate iteration steps in {tag}; select a single uninterrupted run")
        if any(left >= right for left, right in zip(steps, steps[1:])):
            raise ValueError(f"Non-increasing iteration steps in {tag}")
        if reference_steps is None:
            reference_steps = steps
        elif steps != reference_steps:
            raise ValueError(f"Reward iteration steps do not align: {tag}")
        if not all(math.isfinite(event.value) for event in events):
            raise ValueError(f"Non-finite reward values in {tag}")
        series[tag.removeprefix("Episode_Reward/")] = [event.value for event in events]
    missing = {"reaching", "goal_distance"} - series.keys()
    if missing:
        raise ValueError(f"Missing required scalar terms: {sorted(missing)}")
    selected_steps = reference_steps[-last_n:] if last_n else reference_steps
    means = {
        name: math.fsum(values[-last_n:] if last_n else values) / len(selected_steps)
        for name, values in series.items()
    }
    source = read_source_config(log_dir, set(series))
    absolute_total = math.fsum(abs(value) for value in means.values())
    signed_total = math.fsum(means.values())
    positive_total = math.fsum(max(value, 0.0) for value in means.values())
    negative_total = math.fsum(min(value, 0.0) for value in means.values())
    predicted = {
        name: (
            means[name] * source["episode_length_s"] / V1_EPISODE_SECONDS
            * V1_WEIGHTS[name] / source["reward_weights"][name]
        )
        for name in ("reaching", "goal_distance")
    }
    predicted.update({
        "goal_hold": V1_WEIGHTS["goal_hold"] * 5.0 / V1_EPISODE_SECONDS,
        "contact_any": V1_WEIGHTS["contact_any"] * 7.0 / V1_EPISODE_SECONDS,
        "contact_count": V1_WEIGHTS["contact_count"] * 7.0 / V1_EPISODE_SECONDS,
    })
    projection = positive_groups(predicted)
    projection["episode_equivalent_rewards"] = {
        name: value * V1_EPISODE_SECONDS
        for name, value in projection["weighted_logged_means"].items()
    }
    projection["episode_equivalent_positive_sum"] = (
        projection["positive_group_sum"] * V1_EPISODE_SECONDS
    )
    obstacle_raw = 2 * 0.1**2
    obstacle_mean = V1_WEIGHTS["obstacle_displacement"] * min(
        obstacle_raw, OBSTACLE_MAX_SQUARED_DISTANCE
    )
    obstacle_cap = V1_WEIGHTS["obstacle_displacement"] * OBSTACLE_MAX_SQUARED_DISTANCE
    missing_new = sorted(set(V1_NEW_POSITIVE_TERMS) - means.keys())
    return {
        "log_dir": str(log_dir.resolve()),
        "source_configuration": source,
        "records": {
            "available": len(reference_steps),
            "selected": len(selected_steps),
            "requested_last_n": last_n,
            "first_step": selected_steps[0],
            "last_step": selected_steps[-1],
            "validation": "All reward series aligned, strictly increasing, unique and finite.",
        },
        "measured": {
            "weighted_logged_means": means,
            "absolute_mean_sum": absolute_total,
            "share_percent_of_absolute_mean_sum": percentages(
                {name: abs(value) for name, value in means.items()}, absolute_total
            ),
            "signed_mean_sum": signed_total,
            "positive_mean_sum": positive_total,
            "negative_mean_sum": negative_total,
            "penalty_percent_of_positive_mean_sum": (
                -100.0 * negative_total / positive_total if positive_total else None
            ),
        },
        "v1_scenario": {
            "weights": V1_WEIGHTS,
            "episode_length_s": V1_EPISODE_SECONDS,
            "goal_hold_seconds": 5.0,
            "goal_hold_world_xy_radius_m": 0.01,
            "contact_seconds": 7.0,
            "contact_active_count": 5,
            "contact_count_saturation": 5,
            "projection": projection,
            "obstacle": {
                "object_count": 2,
                "displacement_each_m": 0.1,
                "duration_s": 10.0,
                "sum_squared_xyz_displacement_m2": obstacle_raw,
                "max_sum_squared_xyz_displacement_m2": OBSTACLE_MAX_SQUARED_DISTANCE,
                "weighted_logged_mean": obstacle_mean,
                "episode_reward": obstacle_mean * V1_EPISODE_SECONDS,
                "penalty_percent_of_projected_positive_sum": (
                    -100.0 * obstacle_mean / projection["positive_group_sum"]
                ),
                "max_weighted_cost_per_second": -obstacle_cap,
                "max_episode_penalty_magnitude": -obstacle_cap * V1_EPISODE_SECONDS,
            },
        },
        "v1_measured_positive_groups": None if missing_new else positive_groups(means),
        "missing_v1_positive_terms": missing_new,
        "interpretation": [
            "Means equally weight logged iterations, not individual episodes.",
            "Episode_Reward includes weights and dt, divided by maximum episode length.",
            "Absolute shares use abs(mean(term)), not mean(abs(term)).",
            "35/35/20/10 targets use the four positive groups; penalties are reported separately.",
            "Scenario rescales measured progress but assumes 5s goal hold and 7s contact at 5 pads.",
            "Scenario does not predict behavior changes from blind observations or speed thresholds.",
            "Scalar logs cannot reconstruct new reward values from an earlier v0 rollout.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--last-n", type=int, default=0, help="Last N iterations; 0 uses all.")
    parser.add_argument("--output", type=Path, help="Also save the printed JSON to this path.")
    args = parser.parse_args()
    try:
        report = analyze(args.log_dir, args.last_n)
        content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content)
    except (OSError, ValueError) as error:
        print(f"Reward analysis failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(content, end="")


if __name__ == "__main__":
    main()
