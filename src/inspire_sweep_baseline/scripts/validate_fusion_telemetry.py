#!/usr/bin/env python3
"""Validate synchronized Fusion playback CSVs without starting Isaac Sim."""

import argparse
import csv
from contextlib import ExitStack
from itertools import zip_longest
import json
import math
from pathlib import Path


def validate(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["status"] == "complete", "Acquisition did not complete"
    count = metadata["arguments"]["num_envs"]
    steps = metadata["arguments"]["steps"]
    labels = metadata["object_labels"]
    dt = metadata["control_dt_s"]
    assert count > 0 and steps > 0 and dt > 0 and math.isfinite(dt)
    wrench_columns = [f"{a}_{i}" for i in range(count) for a in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
    expected = {
        "wrench_world.csv": ["time", *wrench_columns],
        "wrench_eef.csv": ["time", *wrench_columns],
        "object_velocity.csv": ["time"] + [f"{a}_{obj}_{i}" for i in range(count)
                                                for obj in labels for a in ("vx", "vy", "vz")],
        "episodes.csv": ["time"] + [f"{a}_{i}" for i in range(count) for a in
            ("episode_id", "episode_step", "terminated", "time_out", "delta_valid", "target_id")],
    }
    previous = [None] * count
    terminal_count = invalid_deltas = rows = 0
    with ExitStack() as stack:
        readers = {name: csv.DictReader(stack.enter_context((directory / name).open(newline="")))
                   for name in expected}
        for name, reader in readers.items():
            assert reader.fieldnames == expected[name], f"Unexpected schema: {name}"
            assert len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate columns"
        for batch in zip_longest(*readers.values()):
            assert all(row is not None for row in batch), "CSV files have different row counts"
            rows += 1
            data = {}
            for name, raw in zip(readers, batch):
                assert None not in raw and None not in raw.values(), "Malformed CSV row"
                row = {key: float(value) for key, value in raw.items()}
                assert all(math.isfinite(value) for value in row.values()), "Non-finite CSV sample"
                assert math.isclose(row["time"], rows * dt, rel_tol=1e-10, abs_tol=1e-10), "Time gap or misalignment"
                data[name] = row
            for i in range(count):
                row = {a: data["episodes.csv"][f"{a}_{i}"] for a in
                       ("episode_id", "episode_step", "terminated", "time_out", "delta_valid", "target_id")}
                assert all(value.is_integer() for value in row.values()), "Noninteger episode data"
                assert row["episode_step"] >= 1 and row["episode_id"] >= 0, "Reset sample replaced terminal"
                assert row["target_id"] == metadata["object_names"].index("mass_cylinder")
                assert all(row[k] in (0, 1) for k in ("terminated", "time_out", "delta_valid"))
                last = previous[i]
                if last is None or last["terminated"] or last["time_out"]:
                    assert row["delta_valid"] == 0, "Delta crosses reset boundary"
                    assert row["episode_step"] == 1
                    assert row["episode_id"] == (0 if last is None else last["episode_id"] + 1)
                else:
                    assert row["delta_valid"] == 1
                    assert row["episode_id"] == last["episode_id"]
                    assert row["episode_step"] == last["episode_step"] + 1
                # Rotating axes must preserve force and moment norms.
                for axes in (("Fx", "Fy", "Fz"), ("Mx", "My", "Mz")):
                    norms = [math.sqrt(sum(data[f"wrench_{frame}.csv"][f"{a}_{i}"] ** 2 for a in axes))
                             for frame in ("world", "eef")]
                    assert math.isclose(*norms, rel_tol=2e-5, abs_tol=1e-5), "Wrench frames disagree"
                terminal_count += bool(row["terminated"] or row["time_out"])
                invalid_deltas += row["delta_valid"] == 0
                previous[i] = row
    assert rows == steps == metadata["csv_rows_per_file"], "Incomplete CSV acquisition"
    return dict(passed=True, rows_per_file=rows, columns={k: len(v) for k, v in expected.items()},
                environments=count, objects_per_environment=len(labels),
                seconds_per_environment=rows * dt, terminal_transitions=terminal_count,
                reset_boundary_samples=invalid_deltas,
                checks=["exact schema and finite values", "all files synchronized at control dt",
                        "complete row counts", "terminal and episode boundary continuity",
                        "delta validity does not cross resets", "world/EEF wrench norms agree"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
