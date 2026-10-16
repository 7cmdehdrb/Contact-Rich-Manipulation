"""The independent probe must distinguish misses from invalid poke attempts."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


_MODULE = Path(__file__).resolve().parents[1] / "inspire_tactile_validation" / "reporting.py"
_SPEC = importlib.util.spec_from_file_location("validation_reporting", _MODULE)
reporting = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reporting)


def sample(phase="hold", time_s=0.0, force=0.1, contact=True, channels=(), target_contact=None):
    bits = [index in channels for index in range(17)]
    result = {"phase": phase, "time_s": time_s, "bits": bits, "direct_forces_n": [0.2 if bit else 0.0 for bit in bits], "probe_force_n": force, "probe_contact": contact}
    if target_contact is not None:
        result["probe_target_contact"] = target_contact
    return result


def test_no_contact_does_not_become_false_sensor_miss():
    result = reporting.summarize_trial([sample(force=0.0, contact=False)], 3)
    assert result["status"] == "no_contact"
    assert not result["valid"]
    assert result["target_detected"] is None
    assert result["missed_loaded_contact"] is None
    assert result["detection_fraction"] is None


def test_independent_loaded_contact_with_no_tactile_is_a_miss():
    result = reporting.summarize_trial([sample(target_contact=True)], 3)
    assert result["status"] == "sensor_miss"
    assert result["valid"]
    assert result["missed_loaded_contact"] is True
    assert result["target_contact_verified"]
    assert result["detection_fraction"] == 0.0


def test_wrong_body_contact_is_invalid_not_a_sensor_miss():
    result = reporting.summarize_trial([sample(target_contact=False, channels=(2,))], 3)
    assert result["status"] == "miscontact"
    assert result["loaded_sample_count"] == 1
    assert result["target_loaded_sample_count"] == 0
    assert result["miscontact_sample_count"] == 1
    assert result["missed_loaded_contact"] is None
    assert not any(result["loaded_observed_bits"])


def test_subthreshold_load_is_inconclusive_even_if_bit_active():
    result = reporting.summarize_trial([sample(force=0.049, channels=(3,))], 3)
    assert result["status"] == "insufficient_load"
    assert result["target_detected"] is None
    # Scoring must depend on the independent probe, never direct tactile force.
    assert result["peak_direct_target_force"] == 0.2


def test_loaded_denominator_excludes_baseline_recovery_unloaded_and_miscontact():
    rows = [
        sample("baseline", 0.0, channels=(3, 4)),
        sample("approach", 0.1, force=0.01, channels=(3,)),
        sample("hold", 0.2, channels=(3, 2), target_contact=True),
        sample("hold", 0.3, channels=(2,), target_contact=True),
        sample("approach", 0.4, channels=(3,), target_contact=False),
        sample("recovery", 0.5, force=0.0, contact=False, channels=(4,)),
    ]
    result = reporting.summarize_trial(rows, reporting.CHANNEL_NAMES[3])
    assert result["detection_fraction"] == 0.5
    assert result["loaded_sample_count"] == 3
    assert result["target_loaded_sample_count"] == 2
    assert result["missed_loaded_sample_count"] == 1
    assert result["wrong_channel_indices"] == [2]
    assert result["wrong_channel_bit_max"] == 1.0
    assert result["wrong_channel_max_names"] == [reporting.CHANNEL_NAMES[2]]
    assert result["baseline_false_positive_indices"] == [3, 4]
    assert result["baseline_false_positive_sample_count"] == 1
    assert result["recovery_stuck_indices"] == [4]
    assert result["recovery_probe_contact"] is False


def test_absent_contact_identity_is_reported_as_unverified():
    result = reporting.summarize_trial([sample(channels=(0,))], 0)
    assert result["valid"] and result["target_detected"]
    assert not result["target_contact_verified"]


def test_empty_trial_is_invalid():
    result = reporting.summarize_trial([], 0)
    assert result["status"] == "no_contact"
    assert result["sample_count"] == 0
    assert result["peak_probe_force"] == 0.0


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        reporting.summarize_trial([], 0, threshold)


def test_malformed_channel_vectors_and_times_rejected():
    row = sample()
    row["bits"] = [False]
    with pytest.raises(ValueError, match="17"):
        reporting.summarize_trial([row], 0)
    with pytest.raises(ValueError, match="nondecreasing"):
        reporting.summarize_trial([sample(time_s=1.0), sample(time_s=0.5)], 0)


def test_csv_confusion_uses_trial_any_and_only_valid_trials(tmp_path):
    hit = reporting.summarize_trial([sample(channels=(0, 2)), sample(time_s=0.1)], 0)
    miss = reporting.summarize_trial([sample()], 0)
    invalid = reporting.summarize_trial([sample(contact=False, force=0.0)], 1)
    summary = reporting.write_results(tmp_path, [dict(point="center", **hit), dict(point="corner", **miss), invalid], {"expected_trials": 3})
    assert summary["status"] == "completed"
    assert summary["criteria_status"] == "inconclusive"
    assert summary["valid_loaded_trial_count"] == 2
    assert summary["target_detection_rate_valid_trials"] == 0.5
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with (tmp_path / "confusion.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 17
    assert float(rows[0][reporting.CHANNEL_NAMES[0]]) == 0.5
    assert float(rows[0][reporting.CHANNEL_NAMES[2]]) == 0.5
    assert rows[1][reporting.CHANNEL_NAMES[1]] == ""
    with (tmp_path / "trials.csv").open() as stream:
        trials = list(csv.DictReader(stream))
    assert len(trials) == 3
    assert json.loads(trials[0]["wrong_channel_indices"]) == [2]


def test_clean_pass_and_collection_completion_are_separate(tmp_path):
    hit = reporting.summarize_trial([sample(channels=(0,), target_contact=True)], 0)
    summary = reporting.write_results(tmp_path, [hit], {"expected_trials": 2})
    assert summary["status"] == "incomplete"
    assert summary["criteria_status"] == "pass"
    assert summary["clean_detection_rate_valid_trials"] == 1.0


def test_loaded_miss_fails_criteria(tmp_path):
    miss = reporting.summarize_trial([sample(target_contact=True)], 0)
    summary = reporting.write_results(tmp_path, [miss], {})
    assert summary["criteria_status"] == "fail"
    assert summary["status_counts"] == {"sensor_miss": 1}


def test_retract_only_detection_does_not_hide_loading_miss():
    rows = [sample("hold", 0.0, channels=(), target_contact=True),
            sample("retract", 0.1, channels=(3,), target_contact=True)]
    result = reporting.summarize_trial(rows, 3)
    assert result["status"] == "sensor_miss"
    assert result["detection_fraction"] == 0.0


def test_stationary_hold_force_reference_mismatch_is_inconclusive(tmp_path):
    row = sample(force=2.3e-5, contact=True, channels=(0,), target_contact=True)
    row.update(force_balance_n=0.733, probe_speed_m_s=2e-5)
    result = reporting.summarize_trial([row], 0)
    assert result["status"] == "reference_mismatch"
    assert result["reference_mismatch_count"] == 1
    assert result["max_force_balance_n"] == 0.733
    assert not result["valid"]
    assert result["target_detected"] is None
    assert result["missed_loaded_contact"] is None
    summary = reporting.write_results(tmp_path, [result], {})
    assert summary["criteria_status"] == "inconclusive"
    assert summary["reference_mismatch_sample_count"] == 1
    assert summary["reference_mismatch_trial_count"] == 1


def test_mixed_valid_contact_keeps_detection_but_blocks_clean_pass(tmp_path):
    good = sample(force=0.5, contact=True, channels=(0,), target_contact=True)
    bad = sample(time_s=0.1, force=2e-5, channels=(0,), target_contact=True)
    bad.update(force_balance_n=0.7, probe_speed_m_s=0.0)
    result = reporting.summarize_trial([good, bad], 0)
    assert result["status"] == "detected"
    assert result["valid"]
    assert result["target_detected"] is True
    assert result["reference_mismatch_count"] == 1
    summary = reporting.write_results(tmp_path, [result], {})
    assert summary["criteria_status"] == "inconclusive"
    assert summary["clean_detection_trial_count"] == 0


@pytest.mark.parametrize("phase,speed,balance,probe", [
    ("hold", 0.0, 0.0, 0.0),  # No resistance in free space.
    ("approach", 0.0, 0.7, 0.0),  # Transient approach cannot certify static balance.
    ("hold", 0.001, 0.7, 0.0),  # Speed threshold is strict.
    ("hold", 0.0, 0.049, 0.0),
    ("hold", 0.0, 0.7, 0.025),  # Probe threshold fraction is strict.
])
def test_reference_mismatch_requires_all_conditions(phase, speed, balance, probe):
    row = sample(phase=phase, force=probe)
    row.update(force_balance_n=balance, probe_speed_m_s=speed)
    result = reporting.summarize_trial([row], 0)
    assert result["reference_mismatch_count"] == 0
    assert result["status"] == "insufficient_load"


def test_optional_reference_metrics_do_not_change_old_records():
    result = reporting.summarize_trial([sample(force=0.01)], 0)
    assert result["status"] == "insufficient_load"
    assert result["reference_mismatch_count"] == 0
    assert result["max_force_balance_n"] is None


def test_channel_order_matches_production_geometry_metadata():
    source = Path(__file__).resolve().parents[2] / "inspire_sweep_baseline/inspire_sweep_baseline/assets/data/tactile_regions.json"
    names = tuple(region["name"] for region in json.loads(source.read_text())["regions"])
    assert reporting.CHANNEL_NAMES == names


def test_frame_detection_rate_weights_loaded_frames_not_trials(tmp_path):
    one = reporting.summarize_trial([sample(channels=(0,), target_contact=True)], 0)
    nine = reporting.summarize_trial([
        sample(time_s=float(i), channels=(0,) if i == 0 else (), target_contact=True)
        for i in range(9)
    ], 0)
    summary = reporting.write_results(tmp_path, [one, nine], {})
    assert summary["valid_loaded_sample_count"] == 10
    assert summary["target_detected_loaded_sample_count"] == 2
    assert summary["target_detection_rate_valid_samples"] == pytest.approx(.2)
