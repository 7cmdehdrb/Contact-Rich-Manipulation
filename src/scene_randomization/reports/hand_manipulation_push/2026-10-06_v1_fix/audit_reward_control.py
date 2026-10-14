"""Build the CPU-only, source-backed interpretation of the V1 scalar report.

Run after analyze_logs.py. References point to the saved training source,
so subsequent environment fixes cannot silently change the audit evidence.
"""

from __future__ import annotations

import json
from pathlib import Path


def main():
    out = Path(__file__).resolve().parent
    summary = json.loads((out / "summary.json").read_text())
    run = Path(summary["run"])
    source = run / "source/hand_manipulation_test"

    def cite(relative, needle):
        path = source / relative
        matches = [i + 1 for i, line in enumerate(path.read_text().splitlines()) if needle in line]
        if not matches:
            raise ValueError(f"Missing frozen-source anchor {relative}: {needle}")
        return {"path": str(path), "lines": matches, "anchor": needle}

    last = summary["windows"]["last_100"]
    first = summary["windows"]["first_10"]
    means = last["means"]
    report = {
        "run": str(run),
        "checkpoint": str(run / "model_4300.pt"),
        "analysis_time_utc": summary["analysis_time_utc"],
        "verified_facts": {
            "initial_ten_iterations": {
                "valid_contact_seen_percent": first["means"]["Metrics/target_position/contact_seen"] * 100,
                "raw_contact_seen_percent": first["means"]["Metrics/target_position/raw_contact_seen"] * 100,
                "push_seen_percent": first["means"]["Metrics/target_position/push_seen"] * 100,
                "episodic_progress_contribution": first["episode_reward_contributions"]["progress"],
            },
            "last_hundred_iterations": {
                "valid_contact_seen_percent": means["Metrics/target_position/contact_seen"] * 100,
                "raw_contact_seen_percent": means["Metrics/target_position/raw_contact_seen"] * 100,
                "push_seen_percent": means["Metrics/target_position/push_seen"] * 100,
                "success": means["Metrics/target_position/success"],
                "signed_central_palm_height_error_m": means["Metrics/target_position/palm_height_error_m"],
                "central_palm_approach_gap_m": means["Metrics/target_position/approach_gap_m"],
                "goal_error_m": means["Metrics/target_position/goal_distance_m"],
                "episodic_rewards": last["episode_reward_contributions"],
                "raw_excess_share_of_negative_rewards": last["negative_share"]["action_excess"],
                "terminal_raw_input_axis_clipping_fraction": means["Metrics/target_position/input_action_clip_fraction"],
                "progress_nonzero_iterations": last["nonzero_iterations"]["Episode_Reward/progress"],
                "contact_seen_nonzero_iterations": last["nonzero_iterations"]["Metrics/target_position/contact_seen"],
                "mean_valid_contact_duration_all_selected_episodes_s": means["Metrics/target_position/valid_push_contact_time_s"],
            },
            "all_logged_success_values_zero": summary["all_run_ranges"]["Metrics/target_position/success"]["max"] == 0,
            "training": summary["performance_total"],
            "saved_ppo_unchanged_reference": summary["saved_agent"]["algorithm"],
        },
        "source_findings": [
            {
                "fact": "The 6 cm projection bounds controller target error relative to current C, not absolute EEF height or workspace.",
                "sources": [cite("mdp/push_v1_actions.py", "self._desired_c_pose_b[index, :3] = accumulated_translation_target("),
                            cite("accumulation_math.py", "proposed = previous_target_b + quaternion_rotate("),
                            cite("accumulation_math.py", "return current_position_b + error * scale")],
                "inference": "Repeated upward increments can move the persistent target and measured EEF away from the contact plane while remaining within the per-update error ball.",
            },
            {
                "fact": "Approach shaping rewards new best records only, and is suppressed after first valid contact. Retreat or hovering has no direct approach cost.",
                "sources": [cite("push_state.py", "approach_delta = (approach - self._best_approach).clamp_min(0.)"),
                            cite("push_state.py", "approach_delta *= (~contact_seen & good)"),
                            cite("push_state.py", "self._best_approach[changed] = torch.maximum")],
                "inference": "Once a small approach reward is collected, staying high and avoiding collision can be a local policy attractor. Terminal height/gap statistics support that behavior, but do not by themselves prove the causal trajectory.",
            },
            {
                "fact": "Physical progress requires aligned actual pad-to-Cube force, full-3D palm-normal alignment, same-substep vertical Table support, current support and no failure.",
                "sources": [cite("push_state.py", "active_pair = pair_norm >= threshold"),
                            cite("push_state.py", "aligned_substep &= (palm_dot >= self.cfg.alignment_cos)"),
                            cite("push_state.py", "valid_push = (aligned_substep & support_substep).any(-1) & good"),
                            cite("push_state.py", "progress *= (valid_push & grounded & good)")],
                "inference": "Late raw_contact_seen equals valid contact_seen and is nearly zero, so lack of physical approach dominates the late bottleneck. Raising gated progress weight alone cannot reward episodes that never touch.",
            },
            {
                "fact": "Input excess is a nonpositive raw-Gaussian rate, while physical arm commands clamp each finite input to [-1,1].",
                "sources": [cite("mdp/push_v1_rewards.py", "return -excess.square().mean(-1)"),
                            cite("mdp/push_v1_actions.py", "normalized = sanitized.clamp(-1.0, 1.0)")],
                "inference": "High latent standard deviations produce mostly endpoint physical inputs. This impairs fine motion during stochastic training. Deterministic play uses actor means, so its abnormal motion needs an actual-observation mean-action trace.",
            },
            {
                "fact": "C and H have identical orientation; the initial H/C +X axis is world up. Therefore initial action[0] controls vertical increments, not action[2].",
                "sources": [cite("constants.py", "C_QUAT_H_WXYZ = (1.0, 0.0, 0.0, 0.0)"),
                            cite("push_v1_math.py", "return torch.stack((up, normal, finger), dim=-1)")],
            },
            {
                "fact": "The logged height error refers to the actual central palm reference P, not control point C.",
                "sources": [cite("mdp/push_v1_commands.py", "height_error = sample.palm_pos_w[:, 2]"),
                            cite("mdp/push_v1_events.py", "control = palm + (rotation @ (self.c_offset_h - self.palm_reference_h)")],
            },
        ],
        "recommended_fixes": [
            "Add a V1-only continuous nonpositive actual-C-height soft cost, with explicit metres, deadband and normalization; let RewardManager integrate dt exactly once. Keep reset poses inside the hard envelope.",
            "Treat the actual-C-height hard bound as shared physical failure before reward computation, so positive rewards, success, timeout precedence and failure penalty remain consistent. Add subtype diagnostics for height, robot-Table, footprint, fall and invalid state.",
            "Consider a small ongoing nonpositive pre-contact approach-gap cost if the Z-only correction still leaves horizontal avoidance. Preserve bounded record rewards and real contact/support gates; do not add a stationary positive near-object bonus.",
            "Keep requested reference PPO parameters unchanged. Measure actual-observation raw actor means, processed actions, measured C height and persistent target height during deterministic and stochastic checks before attributing all odd play motion to std.",
        ],
        "limitations": [
            "The run logs only aggregate failure; robot-Table/footprint/fall/invalid failure subtype frequencies cannot be recovered.",
            "Metrics are means of episode-ending selected reset rows, not complete motion trajectories or exact episode-count-weighted population rates.",
            "The 32 microsecond mean valid contact duration includes all selected episodes, including the many with no contact; it is not contact duration conditional on touching.",
            "model_4300.pt has network/optimizer weights and std but no actual 64D rollout observations. The checkpoint report gives analytic stochastic clipping lower bounds, not empirical actor-mean or deterministic-play distributions.",
            "Do not compare differential-entropy coefficients directly to episodic reward magnitudes; PPO losses and reward logs use different normalizations.",
            "A fixed numeric C-height envelope must be checked against realized reset and physical push poses; signed palm height statistics do not certify that envelope.",
        ],
    }
    (out / "source_findings.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(out / "source_findings.json"), "last_100": report["verified_facts"]["last_hundred_iterations"]}, indent=2))


if __name__ == "__main__":
    main()
