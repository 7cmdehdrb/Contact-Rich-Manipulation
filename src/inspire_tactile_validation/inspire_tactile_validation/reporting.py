"""Contact-detection metrics and portable reports; no simulator imports.

An intended hit is scored only when the independent probe reports contact and
load at or above the threshold.  A trajectory that never contacts the hand is
an invalid trial, not evidence that the tactile sensor missed a contact.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CHANNEL_NAMES = (
    "inspire_palm_force_sensor",
    *(f"inspire_thumb_force_sensor_{index}" for index in range(1, 5)),
    *(
        f"inspire_{finger}_force_sensor_{index}"
        for finger in ("index", "middle", "ring", "little")
        for index in range(1, 4)
    ),
)
_ACTIVE_PHASES = frozenset(("approach", "hold"))
_PHASES = _ACTIVE_PHASES | {"baseline", "retract", "recovery"}


def _channel_index(channel: int | str) -> int:
    if isinstance(channel, str):
        try:
            return CHANNEL_NAMES.index(channel)
        except ValueError as error:
            raise ValueError(f"Unknown tactile channel: {channel!r}") from error
    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < len(CHANNEL_NAMES):
        raise ValueError("target_channel must be an index from 0 to 16 or a channel name")
    return channel


def _finite_nonnegative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _bits(value: Iterable[Any]) -> list[bool]:
    result = list(value)
    if len(result) != len(CHANNEL_NAMES) or any(item not in (False, True) for item in result):
        raise ValueError("bits must contain exactly 17 Boolean/0/1 values")
    return [bool(item) for item in result]


def _names(indices: Iterable[int]) -> list[str]:
    return [CHANNEL_NAMES[index] for index in indices]


def summarize_trial(
    samples: Iterable[Mapping[str, Any]], target_channel: int | str, threshold_n: float = 0.05,
) -> dict[str, Any]:
    """Summarize a poke independently of its planned geometric trajectory.

    ``probe_contact`` comes from the probe's contact sensor, not the tactile
    output.  When available, ``probe_target_contact`` additionally identifies
    the intended parent/pad body.  Without that information a loaded hand hit
    is scored, but ``target_contact_verified`` remains false.  Baseline and
    recovery are never counted as intended loaded-contact observations.

    Optional ``force_balance_n`` and ``probe_speed_m_s`` diagnose an unreliable
    force reference: a nearly stationary hold resisting a significant load
    cannot be certified from a probe contact tensor reporting almost zero.
    Such a mismatch never supplies substitute force ground truth for scoring.

    ``detection_fraction`` is sample-based, conditional on loaded intended
    contact.  ``loaded_observed_bits`` records trial-any detections and is the
    basis for the separate trial-based confusion table.  A final recovery bit
    is reported as a residual indication, not proof of a physically stuck
    sensor; continuing probe contact is reported alongside it.
    """
    target = _channel_index(target_channel)
    threshold = _finite_nonnegative(threshold_n, "threshold_n")
    if threshold == 0.0:
        raise ValueError("threshold_n must be positive")
    rows = []
    previous_time = -math.inf
    for sample in samples:
        phase = sample["phase"]
        if phase not in _PHASES:
            raise ValueError(f"Unknown sample phase: {phase!r}")
        time_s = float(sample["time_s"])
        if not math.isfinite(time_s) or time_s < previous_time:
            raise ValueError("time_s must be finite and nondecreasing")
        previous_time = time_s
        forces = list(sample["direct_forces_n"])
        if len(forces) != len(CHANNEL_NAMES):
            raise ValueError("direct_forces_n must contain exactly 17 values")
        rows.append({
            "phase": phase,
            "time_s": time_s,
            "bits": _bits(sample["bits"]),
            "direct_forces_n": [_finite_nonnegative(force, "direct force") for force in forces],
            "probe_force_n": _finite_nonnegative(sample["probe_force_n"], "probe force"),
            "probe_contact": bool(sample["probe_contact"]),
            "probe_target_contact": (
                bool(sample["probe_target_contact"])
                if sample.get("probe_target_contact") is not None else None
            ),
            "force_balance_n": (
                _finite_nonnegative(sample["force_balance_n"], "force balance")
                if sample.get("force_balance_n") is not None else None
            ),
            "probe_speed_m_s": (
                _finite_nonnegative(sample["probe_speed_m_s"], "probe speed")
                if sample.get("probe_speed_m_s") is not None else None
            ),
        })

    active = [row for row in rows if row["phase"] in _ACTIVE_PHASES]
    contact = [row for row in active if row["probe_contact"]]
    loaded = [row for row in contact if row["probe_force_n"] >= threshold]
    intended = [row for row in loaded if row["probe_target_contact"] is not False]
    miscontacts = [row for row in loaded if row["probe_target_contact"] is False]
    observed_counts = [sum(row["bits"][index] for row in intended) for index in range(len(CHANNEL_NAMES))]
    observed_bits = [count > 0 for count in observed_counts]
    wrong_indices = [index for index, detected in enumerate(observed_bits) if index != target and detected]
    wrong_max = max((observed_counts[index] for index in wrong_indices), default=0)
    wrong_max_indices = [index for index in wrong_indices if observed_counts[index] == wrong_max]
    baseline = [row for row in rows if row["phase"] == "baseline"]
    baseline_indices = [index for index in range(len(CHANNEL_NAMES)) if any(row["bits"][index] for row in baseline)]
    recovery = [row for row in rows if row["phase"] == "recovery"]
    recovery_indices = [index for index, bit in enumerate(recovery[-1]["bits"]) if bit] if recovery else []
    reference_mismatches = [
        row for row in rows
        if row["phase"] == "hold"
        and row["probe_speed_m_s"] is not None and row["probe_speed_m_s"] < 0.001
        and row["force_balance_n"] is not None and row["force_balance_n"] >= threshold
        and row["probe_force_n"] < threshold * 0.5
    ]
    valid = bool(intended)
    detected = observed_bits[target] if valid else None
    if valid:
        status = "detected" if detected else "sensor_miss"
    elif reference_mismatches:
        status = "reference_mismatch"
    elif miscontacts:
        status = "miscontact"
    elif contact:
        status = "insufficient_load"
    else:
        status = "no_contact"
    return {
        "target_channel": target,
        "target_channel_name": CHANNEL_NAMES[target],
        "status": status,
        "valid": valid,
        "threshold_n": threshold,
        "sample_count": len(rows),
        "actual_contact_count": len(contact),
        "loaded_sample_count": len(loaded),
        "target_loaded_sample_count": len(intended),
        "miscontact_sample_count": len(miscontacts),
        "reference_mismatch_count": len(reference_mismatches),
        "max_force_balance_n": max((row["force_balance_n"] for row in rows if row["force_balance_n"] is not None), default=None),
        "target_contact_verified": bool(intended) and all(row["probe_target_contact"] is True for row in intended),
        "target_detected": detected,
        "detection_fraction": observed_counts[target] / len(intended) if valid else None,
        "missed_loaded_contact": not detected if valid else None,
        "missed_loaded_sample_count": len(intended) - observed_counts[target],
        "loaded_observed_bits": observed_bits,
        "loaded_detection_fractions": [count / len(intended) for count in observed_counts] if valid else [None] * len(CHANNEL_NAMES),
        "wrong_channel_indices": wrong_indices,
        "wrong_channel_names": _names(wrong_indices),
        "wrong_channel_bit_max": wrong_max / len(intended) if valid else None,
        "wrong_channel_max_indices": wrong_max_indices,
        "wrong_channel_max_names": _names(wrong_max_indices),
        "baseline_sample_count": len(baseline),
        "baseline_false_positive_sample_count": sum(any(row["bits"]) for row in baseline),
        "baseline_false_positive_indices": baseline_indices,
        "baseline_false_positive_names": _names(baseline_indices),
        "recovery_sample_count": len(recovery),
        "recovery_stuck_indices": recovery_indices,
        "recovery_stuck_names": _names(recovery_indices),
        "recovery_probe_contact": recovery[-1]["probe_contact"] if recovery else None,
        "peak_probe_force": max((row["probe_force_n"] for row in rows), default=0.0),
        "peak_direct_target_force": max((row["direct_forces_n"][target] for row in rows), default=0.0),
    }


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) if isinstance(value, (list, dict, tuple)) else value


def write_results(
    output_dir: str | Path, trials: list[dict[str, Any]], metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write trial CSV, trial-any 17x17 confusion rates, and a JSON summary.

    Confusion cells are blank for expected channels without a valid loaded
    trial; they are not fabricated zero detection rates.  The completion
    status describes data collection.  The separate criteria status describes
    this binary check, never physical pressure/force calibration accuracy.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for trial in trials for key in trial))
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in trial.items()} for trial in trials)

    valid = [trial for trial in trials if trial["valid"]]
    channel_trials = {index: [trial for trial in valid if _channel_index(trial["target_channel"]) == index] for index in range(len(CHANNEL_NAMES))}
    with (output / "confusion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["expected_channel", "valid_loaded_trials", *CHANNEL_NAMES])
        for index, name in enumerate(CHANNEL_NAMES):
            group = channel_trials[index]
            rates = [sum(bool(trial["loaded_observed_bits"][column]) for trial in group) / len(group) for column in range(len(CHANNEL_NAMES))] if group else [None] * len(CHANNEL_NAMES)
            writer.writerow([name, len(group), *rates])

    loaded_samples = sum(trial["target_loaded_sample_count"] for trial in valid)
    detected_samples = loaded_samples - sum(trial["missed_loaded_sample_count"] for trial in valid)
    counts = dict(Counter(trial["status"] for trial in trials))
    detected_count = sum(trial["target_detected"] is True for trial in valid)
    clean_count = sum(
        trial["target_detected"] is True
        and not trial["wrong_channel_indices"]
        and not trial["baseline_false_positive_indices"]
        and not trial["recovery_stuck_indices"]
        and not trial.get("reference_mismatch_count", 0)
        for trial in valid
    )
    expected = metadata.get("expected_trials")
    complete = bool(metadata.get("collection_complete", True)) and (expected is None or expected == len(trials))
    mismatch_count = sum(trial.get("reference_mismatch_count", 0) for trial in trials)
    if not valid or len(valid) != len(trials) or mismatch_count:
        criteria_status = "inconclusive"
    else:
        criteria_status = "pass" if clean_count == len(valid) else "fail"
    summary = {
        "status": "completed" if complete else "incomplete",
        "criteria_status": criteria_status,
        "interpretation": "Binary contact detection and channel selectivity check; not physical sensor accuracy or pressure/force calibration.",
        "metadata": dict(metadata),
        "trial_count": len(trials),
        "valid_loaded_trial_count": len(valid),
        "invalid_trial_count": len(trials) - len(valid),
        "valid_loaded_sample_count": loaded_samples,
        "target_detected_loaded_sample_count": detected_samples,
        "target_detection_rate_valid_samples": detected_samples / loaded_samples if loaded_samples else None,
        "reference_mismatch_sample_count": mismatch_count,
        "reference_mismatch_trial_count": sum(trial.get("reference_mismatch_count", 0) > 0 for trial in trials),
        "status_counts": counts,
        "target_detected_trial_count": detected_count,
        "target_detection_rate_valid_trials": detected_count / len(valid) if valid else None,
        "clean_detection_trial_count": clean_count,
        "clean_detection_rate_valid_trials": clean_count / len(valid) if valid else None,
        "criteria": "Clean detection requires an intended loaded hit, target bit detection, no other channel bits during loaded contact, no baseline bits, no final recovery bits, and no independent force-reference mismatch.",
        "confusion_definition": "Trial-any observed bit during independent loaded intended contact; denominator is valid loaded trials for each expected channel.",
        "channel_coverage": {CHANNEL_NAMES[index]: len(group) for index, group in channel_trials.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary
