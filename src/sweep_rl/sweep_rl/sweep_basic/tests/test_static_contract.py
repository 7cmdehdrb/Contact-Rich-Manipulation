"""Static checks for the isolated 31-D, 6-D, single-stage task contract."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1]


def test_no_imports_from_existing_task_implementations():
    forbidden_prefixes = (
        "sweep_rl.osc_sweep",
        "sweep_rl.osc_sweep_independent",
        "sweeping_policy",
    )
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(forbidden_prefixes)
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_prefixes)


def test_scene_entities_and_policy_sensor_contract_are_preserved():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    asset_source = (PACKAGE_ROOT / "assets.py").read_text()
    assert env_source.count("target_object = RigidObjectCfg(") == 1
    assert "RigidObjectCollection" not in env_source
    assert "ContactSensorCfg" not in asset_source
    assert "VirtualFTSensor" not in asset_source
    assert "activate_contact_sensors=False" in asset_source
    assert 'prim_path="{ENV_REGEX_NS}/Shelf/rack"' in env_source
    assert "activate_contact_sensors=True" in env_source
    assert "ROBOT_CONTACT_BODY_PATHS" in env_source
    assert "shelf_contact = ContactSensorCfg(" in env_source
    assert "GRIPPER_CONTACT_BODY_PATHS" in env_source
    assert "target_gripper_contact = ContactSensorCfg(" in env_source


def test_policy_observation_contract_is_noise_free_and_ordered():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    policy_source = env_source.split("class PolicyCfg", 1)[1].split(
        "policy: PolicyCfg", 1
    )[0]
    terms = [
        "joint_pos = ObsTerm",
        "joint_vel = ObsTerm",
        "last_action = ObsTerm",
        "target_object_state = ObsTerm",
        "goal_position = ObsTerm",
        "eef_pose = ObsTerm",
    ]
    indices = [policy_source.index(term) for term in terms]
    assert indices == sorted(indices)
    assert "noise=" not in policy_source
    assert "self.enable_corruption = False" in policy_source
    assert "goal_delta" not in policy_source
    assert "task_phase" not in policy_source
    assert "descent_state" not in policy_source
    assert "target_linear_velocity" not in policy_source
    critic_source = env_source.split("class CriticCfg", 1)[1].split(
        "critic: CriticCfg", 1
    )[0]
    assert "goal_position = ObsTerm" in critic_source
    assert "target_linear_velocity = ObsTerm" in critic_source


def test_action_and_fixed_open_gripper_contract():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    action_source = (PACKAGE_ROOT / "mdp" / "actions.py").read_text()
    assert "OpenGripperRelativeJointPositionActionCfg" in env_source
    assert "scale=0.05" in env_source
    assert "use_default_offset=False" in env_source
    assert "incremental_joint_target" in action_source
    assert "self._asset.data.joint_pos" in action_source
    assert "self._asset.data.soft_joint_pos_limits" in action_source
    assert "gripper_open_position=0.0" in env_source
    assert "def apply_actions" in action_source
    assert "self._gripper_open_targets" in action_source


def test_ppo_horizon_matches_the_long_single_stage_task():
    ppo_source = (PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg.py").read_text()
    assert "num_steps_per_env = 64" in ppo_source
    assert "gamma=0.995" in ppo_source
    assert '"actor": ["policy"]' in ppo_source
    assert '"critic": ["critic"]' in ppo_source


def test_single_stage_fixed_world_y_command_contract():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    command_source = (PACKAGE_ROOT / "mdp" / "commands.py").read_text()
    assert "SWEEP_DISTANCE = 0.40" in env_source
    assert "TARGET_SPEED = 0.10" in env_source
    assert "task_phase" not in command_source
    assert "phase_transition" not in command_source
    assert "descent_enabled" not in command_source
    assert "self.direction_w[:, 1] = 1.0" in command_source
    assert "precontact_point_w" in command_source
    assert "push_point_w" in command_source
    assert "contact_point_pos_w" in command_source
    assert "wrist_pos_w" in command_source
    assert "wrist_frame_index=WRIST_FRAME_INDEX" in env_source
    assert "contact_force_threshold" in command_source
    assert "reach_progress_threshold" not in command_source
    assert "maximum_angle" not in command_source
    update_body = command_source.split("def _update_command", 1)[1].split(
        "def _set_debug_vis_impl", 1
    )[0]
    assert "pass" in update_body


def test_spawn_and_kinematic_shelf_contract():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    assert "TARGET_X_RANGE = (-0.72, -0.68)" in env_source
    assert "TARGET_Y_RANGE = (-0.12, -0.08)" in env_source
    assert "TARGET_HEIGHT = 0.07" in env_source
    assert "TARGET_MASS = 0.50" in env_source
    assert "mass=TARGET_MASS" in env_source
    assert "TARGET_MAXIMUM_HEIGHT_CHANGE = 0.5 * TARGET_HEIGHT" in env_source
    assert '"maximum_height_change": TARGET_MAXIMUM_HEIGHT_CHANGE' in env_source
    assert '"x": (-0.02, 0.02)' in env_source
    assert '"y": (-0.02, 0.02)' in env_source
    assert "kinematic_enabled=True" in env_source
    assert "disable_gravity=True" in env_source
    assert "randomize_target_mass" not in env_source
    assert "randomize_target_friction" not in env_source
    assert "CurriculumCfg" not in env_source


def test_geometry_debug_and_speed_curriculum_contract():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    command_source = (PACKAGE_ROOT / "mdp" / "commands.py").read_text()
    termination_source = (PACKAGE_ROOT / "mdp" / "terminations.py").read_text()
    assert "PRECONTACT_STANDOFF = 0.060" in env_source
    assert "PUSH_STANDOFF = 0.040" in env_source
    assert "EXTERNAL_FINGER_OFFSET = 0.055" in env_source
    assert "external_finger_offset=EXTERNAL_FINGER_OFFSET" in env_source
    assert "SAFE_APPROACH_HEIGHT" not in env_source
    assert "DESCENT_HORIZONTAL_THRESHOLD" not in env_source
    assert "approach_point_w" not in command_source
    assert "contact_frame_index=EEF_FRAME_INDEX" in env_source
    assert "sweep_pad_frame_index" not in env_source
    assert "upper_sweep_pad" not in env_source
    assert "lower_sweep_pad" not in env_source
    assert "SWEEP_BASIC_DEBUG_VIS" in env_source
    assert "_debug_vis_callback" in command_source
    assert "initial_speed\": 0.30" in env_source
    assert "intermediate_speed\": 0.20" in env_source
    assert "final_speed\": 0.15" in env_source
    assert "sustained_steps\": 3" in env_source
    assert "update_sustained_violation" in termination_source


def test_reward_and_termination_contract_is_exact():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    reward_section = env_source.split("class RewardsCfg", 1)[1].split(
        "class TerminationsCfg", 1
    )[0]
    termination_section = env_source.split("class TerminationsCfg", 1)[1].split(
        "class SweepBasicEnvCfg", 1
    )[0]
    reward_names = [
        "action_rate_l2",
        "joint_vel_l2",
        "shelf_collision",
        "reward_for_hand_reaching",
        "align_ee_target",
        "pushing_target",
        "target_contact",
        "contact_forward_progress",
        "velocity_tracking",
        "endpoint_error",
        "stopped_at_goal",
        "sweep_success",
        "lateral_error",
        "overshoot",
        "stall",
        "midpoint_no_entry",
        "shelf_collision_failure",
    ]
    termination_names = [
        "time_out",
        "success",
        "object_drop",
        "push_fast",
        "shelf_collision",
    ]
    assert reward_section.count("= RewTerm(") == len(reward_names)
    assert all(f"{name} = RewTerm(" in reward_section for name in reward_names)
    assert termination_section.count("= DoneTerm(") == len(termination_names)
    assert all(f"{name} = DoneTerm(" in termination_section for name in termination_names)
    assert "func=mdp.target_reached_and_stopped" in termination_section
    assert "self.episode_length_s = 15.0" in env_source
    assert '"sensor_name": "shelf_contact"' in termination_section
    assert '"force_threshold": 0.1' in termination_section
    assert "func=mdp.remaining_horizon_failure_penalty" in reward_section
    assert '"term_names": ("shelf_collision",)' in reward_section
    assert 'weight=-8.0' in reward_section
    assert "func=mdp.MidpointNoEntryPenalty" in reward_section
    assert '"contact_dwell_time": 0.10' in reward_section
    assert '"episode_fraction": 0.50' in reward_section
    assert 'weight=-10.0' in reward_section
    assert "weight=-0.01" in reward_section
    assert "func=mdp.align_ee_target" in reward_section
    assert "weight=2.0" in reward_section
    assert "func=mdp.pushing_target" in reward_section
    assert "weight=6.0" in reward_section
    assert '"distance_scale": 0.18' in reward_section
    reward_source = (PACKAGE_ROOT / "mdp" / "rewards.py").read_text()
    assert "self._contact_elapsed[env_ids] = 0.0" in reward_source
    assert "self._entered[env_ids] = False" in reward_source


def test_all_dense_push_rewards_are_single_stage():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    reward_source = (PACKAGE_ROOT / "mdp" / "rewards.py").read_text()
    assert '"distance_std": 0.25' in env_source
    assert "command.push_point_w()" in reward_source
    assert "command.wrist_pos_w()" in reward_source
    assert "signed_square_alignment_score" in reward_source
    assert "task_phase" not in reward_source
    assert "phase_transition" not in reward_source
    assert "transition_bonus" not in reward_source
    for reward_name in (
        "target_contact_bonus",
        "contact_forward_progress",
        "object_velocity_tracking",
        "normalized_endpoint_error",
        "stopped_at_goal",
        "sweep_success_bonus",
        "lateral_displacement",
        "overshoot_penalty",
        "object_stall_penalty",
    ):
        assert f"func=mdp.{reward_name}" in env_source


def test_no_virtual_contact_pad_or_pad_observation_is_added():
    env_source = (PACKAGE_ROOT / "env_cfg.py").read_text()
    asset_source = (PACKAGE_ROOT / "assets.py").read_text()
    observations = env_source.split("class ObservationsCfg", 1)[1].split(
        "class EventsCfg", 1
    )[0]
    assert "LeftSweepContactPad" not in env_source + asset_source
    assert "RightSweepContactPad" not in env_source + asset_source
    assert "contact_point" not in observations
    assert "contact_force" not in observations
