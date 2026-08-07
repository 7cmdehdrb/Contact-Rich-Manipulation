"""Static contract checks for the Sweep-Policy Cube port."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1]
ENV_SOURCE = (PACKAGE / "env_cfg.py").read_text()
REWARD_SOURCE = (PACKAGE / "mdp" / "rewards.py").read_text()
AGENT_SOURCE = (PACKAGE / "agents" / "rsl_rl_ppo_cfg.py").read_text()
ROOT_INIT_SOURCE = (PACKAGE.parent / "__init__.py").read_text()


def _class_assignments(class_name: str) -> set[str]:
    module = ast.parse(ENV_SOURCE)
    class_node = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        target.id
        for node in class_node.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }


def _reward_weight(term_name: str) -> float:
    module = ast.parse(ENV_SOURCE)
    rewards = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg"
    )
    assignment = next(
        node
        for node in rewards.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == term_name
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    weight = next(
        keyword.value
        for keyword in assignment.value.keywords
        if keyword.arg == "weight"
    )
    value = ast.literal_eval(weight)
    assert isinstance(value, (int, float))
    return float(value)


def test_registration_and_root_import_contract():
    registration = (PACKAGE / "__init__.py").read_text()
    assert "Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0" in registration
    assert "SweepPolicyCubeEnvCfg" in registration
    assert "from . import sweep_policy_cube" in ROOT_INIT_SOURCE


def test_single_cube_physics_contract():
    assert "size=(TARGET_SIDE, TARGET_SIDE, TARGET_HEIGHT)" in ENV_SOURCE
    assert "TARGET_SIDE = 0.06" in ENV_SOURCE
    assert "TARGET_HEIGHT = 0.15" in ENV_SOURCE
    assert (
        "TARGET_CENTER_HEIGHT = SHELF_SURFACE_HEIGHT + 0.5 * TARGET_HEIGHT"
        in ENV_SOURCE
    )
    assert "TARGET_MASS = 0.50" in ENV_SOURCE
    assert "SHELF_POSITION = (-0.70, 0.0, 0.0)" in ENV_SOURCE
    assert "(-0.75, -0.20, TARGET_CENTER_HEIGHT)" in ENV_SOURCE
    assert "(-0.75, 0.00, TARGET_CENTER_HEIGHT)" in ENV_SOURCE
    assert "(-0.60, -0.20, TARGET_CENTER_HEIGHT)" in ENV_SOURCE
    assert "(-0.60, 0.00, TARGET_CENTER_HEIGHT)" in ENV_SOURCE
    assert "(-0.75, 0.20, TARGET_CENTER_HEIGHT)" not in ENV_SOURCE
    assert "(-0.60, 0.20, TARGET_CENTER_HEIGHT)" not in ENV_SOURCE
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=TARGET_MASS)" in ENV_SOURCE
    assert _class_assignments("SweepPolicyCubeSceneCfg") >= {
        "ground",
        "shelf",
        "robot",
        "ee_frame",
        "finger_frame",
        "wrist_frame",
        "target_object",
        "light",
    }
    assert "RigidObjectCollection" not in ENV_SOURCE


def test_fixed_goal_width_and_observation_contract():
    assert "GOAL_OFFSET = (0.0, 0.18, 0.0)" in ENV_SOURCE
    assert 'params={"width": TARGET_SIDE}' in ENV_SOURCE
    policy_terms = _class_assignments("PolicyCfg")
    assert policy_terms == {
        "joint_pos",
        "joint_vel",
        "actions",
        "target_obs_state",
        "target_obj_width",
        "ee_pose",
        "goal_pos",
    }
    assert "Unoise" not in ENV_SOURCE
    assert "self.enable_corruption = False" in ENV_SOURCE
    observation_source = (PACKAGE / "mdp" / "observations.py").read_text()
    assert "def joint_pos_rel_arm(" in observation_source
    assert "robot.data.joint_pos[:, :6]" in observation_source
    assert "joint_pos_rel_eight" not in ENV_SOURCE + observation_source


def test_no_multi_object_or_object_collision_reward_contract():
    active_source = ENV_SOURCE + REWARD_SOURCE
    for forbidden in (
        "object_collision",
        "target_id",
        "load_yaml_config",
        "environment.yaml",
        "isaaclab_tasks.manager_based.manipulation.shelf",
        "example.Sweep-Policy",
    ):
        assert forbidden not in active_source
    assert "env.sweep_dir" not in active_source
    assert _class_assignments("RewardsCfg") == {
        "action_rate",
        "joint_vel",
        "end_effector_position_tracking",
        "end_effector_position_tracking_fine_grained",
        "orientation",
    }
    assert _class_assignments("TerminationsCfg") == {"time_out"}
    assert "curriculum = None" in ENV_SOURCE


def test_arm_only_action_and_timing_contract():
    assert "scale=0.5" in ENV_SOURCE
    assert "JointPositionActionCfg" in ENV_SOURCE
    assert "BinaryJointPositionActionCfg" not in ENV_SOURCE
    assert _class_assignments("ActionsCfg") == {"arm_action"}
    assert "stiffness=2000.0" in ENV_SOURCE
    assert "damping=1000.0" in ENV_SOURCE
    assert '".*(finger|knuckle).*": 0.0' in ENV_SOURCE
    assert "self.episode_length_s = 10.0" in ENV_SOURCE
    assert "self.decimation = 2" in ENV_SOURCE
    assert "self.sim.dt = 0.01" in ENV_SOURCE


def test_physical_finger_tcp_and_training_balance_contract():
    assert 'GRIPPER_BASE_BODY_PATH = "robotiq_base_link"' in ENV_SOURCE
    assert "EE_FORWARD_OFFSET = 0.130" in ENV_SOURCE
    assert "FINGER_CONTACT_LATERAL_OFFSET = 0.070" in ENV_SOURCE
    assert "WRIST_BACK_OFFSET = -0.140" in ENV_SOURCE
    assert "OffsetCfg(pos=(EE_FORWARD_OFFSET, 0.0, 0.0))" in ENV_SOURCE
    assert "OffsetCfg(pos=(WRIST_BACK_OFFSET, 0.0, 0.0))" in ENV_SOURCE
    assert _reward_weight("end_effector_position_tracking") == -0.2
    assert _reward_weight("end_effector_position_tracking_fine_grained") == 0.1
    assert _reward_weight("action_rate") == -0.001
    assert _reward_weight("joint_vel") == -0.0001
    assert _reward_weight("orientation") == -0.1
    assert "func=mdp.ee_y_shelf_z_orientation_error" in ENV_SOURCE
    assert "axis_alignment_error_kernel" in REWARD_SOURCE
    assert "ee_y_shelf_z_alignment_kernel" in REWARD_SOURCE
    assert "align_ee_target" not in ENV_SOURCE + REWARD_SOURCE
    assert "# shelf_collision = RewTerm" in ENV_SOURCE
    assert "# reaching = RewTerm" in ENV_SOURCE
    assert "# sweeping_object = RewTerm" in ENV_SOURCE
    assert "# homing_after_sweep = RewTerm" in ENV_SOURCE
    assert "# object_drop = DoneTerm" in ENV_SOURCE
    assert "# push_fast = DoneTerm" in ENV_SOURCE
    assert "# shelf_collision = DoneTerm" in ENV_SOURCE
    assert "# hand_velocity = DoneTerm" in ENV_SOURCE
    assert "shelf_entry_penalty" not in ENV_SOURCE + REWARD_SOURCE
    assert "early_failure_escape_penalty" not in ENV_SOURCE + REWARD_SOURCE
    assert "_target_offset" not in REWARD_SOURCE
    assert "fixed_policy_hand_target" in REWARD_SOURCE
    assert "reaching_position_command_error" in REWARD_SOURCE
    assert "reaching_position_command_error_tanh" in REWARD_SOURCE
    assert "position_error_tanh_kernel" in REWARD_SOURCE
    assert "return torch.exp(" in REWARD_SOURCE
    assert "torch.sign(sweep_direction_y)" in (
        PACKAGE / "mdp" / "reward_math.py"
    ).read_text()
    reward_math_source = (PACKAGE / "mdp" / "reward_math.py").read_text()
    assert "hand_target[:, 0]" not in reward_math_source
    assert "hand_target[:, 2] += 0.03" in reward_math_source
    assert "ee_frame.data.target_pos_w[:, 0]" in REWARD_SOURCE
    assert "wrist_y_error < 0.04" in (PACKAGE / "mdp" / "reward_math.py").read_text()
    assert "wrist_behind_distance" not in REWARD_SOURCE
    assert "+ 1.06" not in REWARD_SOURCE


def test_ppo_stability_contract():
    assert "clip_actions" not in AGENT_SOURCE
    assert "init_std=0.25" in AGENT_SOURCE
    assert "entropy_coef=0.001" in AGENT_SOURCE
    assert "gamma=0.99" in AGENT_SOURCE


def test_original_combined_robot_asset_and_joint_contract():
    assert "DEFAULT_SWEEP_POLICY_ROBOT_USD_PATH" in ENV_SOURCE
    assert "Collected_UR5e_v4/UR5e_v4.usd" in ENV_SOURCE
    assert '"SWEEP_POLICY_ROBOT_USD_PATH"' in ENV_SOURCE
    assert "make_sensorless_robot_cfg" not in ENV_SOURCE
    for joint_name in (
        "finger_joint",
        "right_outer_knuckle_joint",
        "left_outer_finger_joint",
        "left_inner_finger_knuckle_joint",
        "left_inner_finger_joint",
        "right_outer_finger_joint",
        "right_inner_finger_joint",
        "right_inner_finger_knuckle_joint",
    ):
        assert f'"{joint_name}"' in ENV_SOURCE
