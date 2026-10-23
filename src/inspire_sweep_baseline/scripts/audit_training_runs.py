"""Read-only audit of TensorBoard history and every saved checkpoint in a tree."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def nonfinite_paths(value, path=""):
    if torch.is_tensor(value):
        return [path] if not bool(torch.isfinite(value).all()) else []
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, (list, tuple)) else []
    return [bad for key, item in items for bad in nonfinite_paths(item, f"{path}/{key}")]


def metric(events):
    finite = [event for event in events if math.isfinite(event.value)]
    pair = lambda event: [event.step, event.value if math.isfinite(event.value) else str(event.value)]
    return {
        "count": len(events), "first": pair(events[0]), "last": pair(events[-1]),
        "peak_absolute": pair(max(finite, key=lambda event: abs(event.value))) if finite else None,
        "nonfinite": [pair(event) for event in events if not math.isfinite(event.value)],
    }


def audit_run(run):
    params = run / "params" / "agent.yaml"
    config = yaml.load(params.read_text(encoding="utf-8"), Loader=yaml.BaseLoader) if params.exists() else {}
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    scalars = {tag: metric(accumulator.Scalars(tag)) for tag in accumulator.Tags()["scalars"]
               if accumulator.Scalars(tag)}
    checkpoints = []
    for path in sorted(run.glob("model_*.pt"), key=lambda item: int(item.stem.split("_")[-1])):
        state = torch.load(path, map_location="cpu", weights_only=False)
        noise = {key: value.tolist() for key, value in state.get("actor_state_dict", {}).items() if "std" in key}
        checkpoints.append({"file": path.name, "iteration": state.get("iter"), "noise": noise,
                            "nonfinite_tensors": nonfinite_paths(state)})
    return {"run": str(run), "agent_config": config, "metrics": scalars, "checkpoints": checkpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Write reports outside the training log tree.")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("--output must be outside the audited log tree")
    runs = sorted({path.parent for path in args.root.rglob("events.out.tfevents.*")}
                  | {path.parent for path in args.root.rglob("model_*.pt")})
    results = [audit_run(run) for run in runs]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "first_iteration", "last_iteration", "peak_value_loss", "nonfinite_loss_count",
                         "checkpoints", "bad_checkpoint_count"])
        for result in results:
            loss = result["metrics"].get("Loss/value", {})
            row = [result["run"], loss.get("first", [None])[0], loss.get("last", [None])[0],
                   loss.get("peak_absolute", [None, None])[1], len(loss.get("nonfinite", [])),
                   len(result["checkpoints"]), sum(bool(c["nonfinite_tensors"]) for c in result["checkpoints"])]
            writer.writerow(row)
            print(row)


if __name__ == "__main__":
    main()
