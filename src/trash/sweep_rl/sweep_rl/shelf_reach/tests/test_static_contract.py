"""Simulator-independent contracts for the shelf-reach environment."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1]
ENV_SOURCE = (PACKAGE / "env_cfg.py").read_text()
INIT_SOURCE = (PACKAGE / "__init__.py").read_text()
ROOT_INIT_SOURCE = (PACKAGE.parent / "__init__.py").read_text()
COMMAND_SOURCE = (PACKAGE / "mdp" / "commands.py").read_text()
REWARD_SOURCE = (PACKAGE / "mdp" / "rewards.py").read_text()
AGENT_SOURCE = (PACKAGE / "agents" / "rsl_rl_ppo_cfg.py").read_text()


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
    return float(ast.literal_eval(weight))


def test_registration_contract():
    assert "Isaac-Reach-Shelf-UR5e-Gripper-v0" in INIT_SOURCE
    assert "Isaac-Reach-Shelf-UR5e-Gripper-Play-v0" in INIT_SOURCE
    assert "UR5eGripperShelfReachEnvCfg" in INIT_SOURCE
    assert "UR5eGripperShelfReachEnvCfg_PLAY" in INIT_SOURCE
    assert "from . import shelf_reach" in ROOT_INIT_SOURCE


def test_scene_has_no_physical_target_contract():
    assert _class_assignments("ShelfReachSceneCfg") == {
        "ground",
        "shelf",
        "robot",
        "ee_frame",
        "light",
    }
    assert "RigidObjectCollectionCfg" not in ENV_SOURCE
    assert "target_object" not in ENV_SOURCE
    assert "SHELF_POSITION = (-0.7, 0.0, 0.0)" in ENV_SOURCE
    assert "mass=100.0" in ENV_SOURCE
    assert "Collected_UR5e_v4/UR5e_v4.usd" in ENV_SOURCE


def test_tcp_and_virtual_command_contract():
    assert 'GRIPPER_BASE_BODY_PATH = "robotiq_base_link"' in ENV_SOURCE
    assert "TCP_FORWARD_OFFSET = 0.13" in ENV_SOURCE
    assert "OffsetCfg(pos=(TCP_FORWARD_OFFSET, 0.0, 0.0))" in ENV_SOURCE
    assert "TARGET_POS_X = (0.55, 0.75)" in ENV_SOURCE
    assert "TARGET_POS_Y = (-0.20, 0.20)" in ENV_SOURCE
    assert "TARGET_POS_Z = (0.30, 0.50)" in ENV_SOURCE
    assert "TARGET_ROLL = (math.pi / 2.0, math.pi / 2.0)" in ENV_SOURCE
    assert "TARGET_PITCH = (0.0, 0.0)" in ENV_SOURCE
    assert "TARGET_YAW = (0.0, 0.0)" in ENV_SOURCE
    assert "resampling_time_range=(4.0, 4.0)" in ENV_SOURCE
    assert "TcpUniformPoseCommand" in COMMAND_SOURCE
    assert "tcp_frame.data.target_pos_w" in COMMAND_SOURCE
    assert "tcp_frame.data.target_quat_w" in COMMAND_SOURCE


def test_arm_only_observation_action_and_reset_contract():
    assert _class_assignments("PolicyCfg") == {
        "joint_pos",
        "joint_vel",
        "pose_command",
        "actions",
    }
    assert _class_assignments("ActionsCfg") == {"arm_action"}
    assert "JointPositionActionCfg" in ENV_SOURCE
    assert "BinaryJointPositionActionCfg" not in ENV_SOURCE
    assert "scale=0.5" in ENV_SOURCE
    assert '"asset_cfg": ARM_CFG' in ENV_SOURCE
    assert _class_assignments("EventCfg") == {"reset_all", "reset_robot_joints"}
    assert "reset_scene_to_default" in ENV_SOURCE
    assert '"reset_joint_targets": True' in ENV_SOURCE
    assert '"position_range": (0.75, 1.25)' in ENV_SOURCE
    assert '".*(finger|knuckle).*": 0.0' in ENV_SOURCE
    assert "stiffness=2000.0" in ENV_SOURCE
    assert "damping=1000.0" in ENV_SOURCE


def test_reach_reward_timing_and_curriculum_contract():
    assert _class_assignments("RewardsCfg") == {
        "end_effector_position_tracking",
        "end_effector_position_tracking_fine_grained",
        "end_effector_orientation_tracking",
        "action_rate",
        "joint_vel",
    }
    assert _reward_weight("end_effector_position_tracking") == -0.2
    assert _reward_weight("end_effector_position_tracking_fine_grained") == 0.1
    assert _reward_weight("end_effector_orientation_tracking") == -0.1
    assert _reward_weight("action_rate") == -0.0001
    assert _reward_weight("joint_vel") == -0.0001
    assert "quat_error_magnitude" in REWARD_SOURCE
    assert _class_assignments("TerminationsCfg") == {"time_out"}
    assert _class_assignments("CurriculumCfg") == {"action_rate", "joint_vel"}
    assert '"weight": -0.005' in ENV_SOURCE
    assert '"weight": -0.001' in ENV_SOURCE
    assert ENV_SOURCE.count('"num_steps": 4500') == 2
    assert "self.decimation = 2" in ENV_SOURCE
    assert "self.episode_length_s = 12.0" in ENV_SOURCE
    assert "self.sim.dt = 1.0 / 60.0" in ENV_SOURCE
    assert "self.commands.ee_pose.debug_vis = True" in ENV_SOURCE


def test_ppo_contract():
    assert "experiment_name = \"reach_shelf_ur5e_gripper\"" in AGENT_SOURCE
    assert "num_steps_per_env = 24" in AGENT_SOURCE
    assert "max_iterations = 1_000" in AGENT_SOURCE
    assert AGENT_SOURCE.count("hidden_dims=[64, 64]") == 2
    assert "init_std=1.0" in AGENT_SOURCE
    assert "entropy_coef=0.01" in AGENT_SOURCE
    assert "gamma=0.99" in AGENT_SOURCE
    assert "desired_kl=0.01" in AGENT_SOURCE
