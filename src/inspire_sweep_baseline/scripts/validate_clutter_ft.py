#!/usr/bin/env python3
"""Validate a completed physical F/T acquisition without Isaac Sim or Torch."""

import argparse
import csv
import json
import math
from pathlib import Path

AXES = ("fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm")


def validate(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["status"] == "complete", "Acquisition did not complete"
    count = metadata["arguments"]["num_envs"]
    steps = metadata["arguments"]["steps"]
    previous = {}
    terminal_count = 0
    tactile_rows = 0
    goal_rows = 0
    rows = 0
    max_identity_error = 0.0
    force_max = 0.0
    moment_max = 0.0
    max_delta_force = 0.0
    max_delta_moment = 0.0
    with (directory / "wrench.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        assert len(columns) == len(set(columns)), "Duplicate CSV columns"
        reward_columns = [c for c in columns if c.startswith("reward_") and c not in ("reward_net", "reward_positive")]
        for raw in reader:
            row = {key: float(value) for key, value in raw.items()}
            assert all(math.isfinite(value) for value in row.values()), f"Nonfinite row {rows}"
            env_id = int(row["env_id"])
            assert env_id == rows % count, "Missing, duplicate, or unordered environment row"
            assert row["control_step"] == rows // count + 1, "Missing/duplicate control step"
            assert row["sample_valid"] == 1, "Stale/reset sample included"
            assert row["episode_step"] >= 1, "Terminal sample overwritten by reset"
            assert row["learning_iteration"] == metadata["checkpoint_iteration"], "Curriculum advanced during replay"
            for frame in ("eef", "world"):
                for axis in AXES:
                    raw_value = row[f"v2_raw_{frame}_{axis}"]
                    reconstructed = row[f"v2_support_{frame}_{axis}"] + row[f"v2_pure_{frame}_{axis}"]
                    error = abs(raw_value - reconstructed)
                    max_identity_error = max(max_identity_error, error)
                    assert math.isclose(raw_value, reconstructed, abs_tol=1e-4, rel_tol=1e-5), "raw != support + pure"
            assert math.isclose(sum(row[c] for c in reward_columns), row["reward_net"], abs_tol=1e-5, rel_tol=1e-5)
            assert math.isclose(sum(max(row[c], 0) for c in reward_columns), row["reward_positive"], abs_tol=1e-5, rel_tol=1e-5)
            last = previous.get(env_id)
            if last is None or last["terminated"] or last["time_out"]:
                assert row["delta_valid"] == 0, "Delta crosses episode boundary"
                if last is not None:
                    assert row["episode_id"] == last["episode_id"] + 1
                    assert row["episode_step"] == 1
            else:
                assert row["delta_valid"] == 1
                assert row["episode_id"] == last["episode_id"]
                assert row["episode_step"] == last["episode_step"] + 1
                for axis in AXES:
                    expected = row[f"v2_pure_world_{axis}"] - last[f"v2_pure_world_{axis}"]
                    assert math.isclose(row[f"delta_pure_world_{axis}"], expected, abs_tol=1e-5, rel_tol=1e-5)
            if row["delta_valid"] == 0:
                assert row["delta_force_norm_N"] == row["delta_moment_norm_Nm"] == 0
            terminal_count += bool(row["terminated"] or row["time_out"])
            tactile_rows += row["tactile_count"] > 0
            goal_rows += row["inside_goal_1cm"] > 0
            force_max = max(force_max, math.sqrt(sum(row[f"v2_pure_world_{a}"]**2 for a in AXES[:3])))
            moment_max = max(moment_max, math.sqrt(sum(row[f"v2_pure_world_{a}"]**2 for a in AXES[3:])))
            max_delta_force = max(max_delta_force, row["delta_force_norm_N"])
            max_delta_moment = max(max_delta_moment, row["delta_moment_norm_Nm"])
            previous[env_id] = row
            rows += 1
    assert rows == steps * count == metadata["csv_rows"], "Incomplete CSV"
    return dict(
        passed=True, rows=rows, columns=len(columns), environments=count, control_steps=steps,
        terminal_transitions=terminal_count, tactile_positive_rows=tactile_rows, inside_goal_rows=goal_rows,
        max_raw_minus_support_minus_pure_error=max_identity_error,
        pure_force_max_norm_N=force_max, pure_moment_max_norm_Nm=moment_max,
        force_change_max_norm_N=max_delta_force, moment_change_max_norm_Nm=max_delta_moment,
        checks=["all numeric values finite", "complete unique ordered env/step rows", "all physical samples fresh",
                "terminal transitions retained", "delta history does not cross resets",
                "v2 raw = support + pure in EEF and world axes", "reward components sum to net",
                "checkpoint curriculum iteration fixed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
