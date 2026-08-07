"""Static contracts for the inherited Cube pre-reach environment."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1]
ENV_SOURCE = (PACKAGE / "env_cfg.py").read_text()
COMMAND_SOURCE = (PACKAGE / "mdp" / "commands.py").read_text()
OBS_SOURCE = (PACKAGE / "mdp" / "observations.py").read_text()
REWARD_SOURCE = (PACKAGE / "mdp" / "rewards.py").read_text()
INIT_SOURCE = (PACKAGE / "__init__.py").read_text()
ROOT_INIT_SOURCE = (PACKAGE.parent / "__init__.py").read_text()
AGENT_SOURCE = (PACKAGE / "agents" / "rsl_rl_ppo_cfg.py").read_text()


def _class_node(class_name: str) -> ast.ClassDef:
    module = ast.parse(ENV_SOURCE)
    return next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _class_assignments(class_name: str) -> set[str]:
    return {
        target.id
        for node in _class_node(class_name).body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }


def test_registration_and_inheritance_contract():
    assert "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0" in INIT_SOURCE
    assert "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0" in INIT_SOURCE
    assert "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1" in INIT_SOURCE
    assert "Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v1" in INIT_SOURCE
    assert "from . import shelf_cube_pre_reach" in ROOT_INIT_SOURCE
    env_class = _class_node("UR5eGripperShelfCubePreReachEnvCfg")
    assert any(
        isinstance(base, ast.Name) and base.id == "UR5eGripperShelfReachEnvCfg"
        for base in env_class.bases
    )
    scene_class = _class_node("ShelfCubePreReachSceneCfg")
    assert any(
        isinstance(base, ast.Name) and base.id == "ShelfReachSceneCfg"
        for base in scene_class.bases
    )
    v1_env_class = _class_node("UR5eGripperShelfCubePreReachEnvCfgV1")
    assert any(
        isinstance(base, ast.Name)
        and base.id == "UR5eGripperShelfCubePreReachEnvCfg"
        for base in v1_env_class.bases
    )


def test_cube_physics_and_safe_spawn_contract():
    assert "CUBE_WIDTH = 0.06" in ENV_SOURCE
    assert "CUBE_DEPTH = 0.06" in ENV_SOURCE
    assert "CUBE_HEIGHT = 0.15" in ENV_SOURCE
    assert "CUBE_MASS = 0.50" in ENV_SOURCE
    assert "CUBE_CENTER_HEIGHT = SHELF_SURFACE_HEIGHT + 0.5 * CUBE_HEIGHT" in ENV_SOURCE
    assert "CUBE_INITIAL_POSITION = (-0.70, -0.10, CUBE_CENTER_HEIGHT)" in ENV_SOURCE
    assert "size=(CUBE_WIDTH, CUBE_DEPTH, CUBE_HEIGHT)" in ENV_SOURCE
    assert "mass=sim_utils.MassPropertiesCfg" not in ENV_SOURCE
    assert "mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS)" in ENV_SOURCE
    assert _class_assignments("ShelfCubePreReachSceneCfg") == {
        "shelf_floor_contact",
        "target_object",
    }


def test_cube_relative_goal_contract():
    assert "BEHIND_WIDTH_SCALE = 1.2" in ENV_SOURCE
    assert "PRE_REACH_Z_OFFSET = 0.03" in ENV_SOURCE
    assert "CubePreReachPoseCommand(TcpUniformPoseCommand)" in COMMAND_SOURCE
    assert "-self.cfg.cube_width * self.cfg.behind_width_scale" in COMMAND_SOURCE
    assert "self.cfg.z_offset" in COMMAND_SOURCE
    assert "self.target.data.root_pos_w" in COMMAND_SOURCE
    assert "subtract_frame_transforms" in COMMAND_SOURCE
    assert "target_roll=TARGET_ROLL[0]" in ENV_SOURCE
    assert "target_pitch=TARGET_PITCH[0]" in ENV_SOURCE
    assert "target_yaw=TARGET_YAW[0]" in ENV_SOURCE


def test_29d_observation_contract():
    assert _class_assignments("PolicyCfg") == {
        "joint_pos",
        "joint_vel",
        "pose_command",
        "target_object_position",
        "cube_width",
        "actions",
    }
    assert "def target_object_position_b(" in OBS_SOURCE
    assert "target.data.root_pos_w" in OBS_SOURCE
    assert "def cube_width(" in OBS_SOURCE
    assert 'params={"width": CUBE_WIDTH}' in ENV_SOURCE


def test_shelf_collision_penalty_contract():
    assert "class CubePreReachRewardsCfg(RewardsCfg)" in ENV_SOURCE
    assert "SHELF_CONTACT_FORCE_THRESHOLD = 1.0" in ENV_SOURCE
    assert "SHELF_COLLISION_WEIGHT = -0.02" in ENV_SOURCE
    assert "self.shelf.spawn.activate_contact_sensors = True" in ENV_SOURCE
    assert "shelf_floor_contact = ContactSensorCfg(" in ENV_SOURCE
    assert (
        'prim_path=f"{{ENV_REGEX_NS}}/Shelf/{SHELF_FLOOR_BODY_PATH}"'
        in ENV_SOURCE
    )
    assert "track_contact_points=True" in ENV_SOURCE
    assert "for body_path in ROBOT_CONTACT_BODY_PATHS" in ENV_SOURCE
    assert '"base_link"' in ENV_SOURCE
    assert '"wrist_3_link"' in ENV_SOURCE
    assert '"robotiq_base_link"' in ENV_SOURCE
    assert '"left_inner_knuckle"' in ENV_SOURCE
    assert '"right_inner_knuckle"' in ENV_SOURCE
    assert '"TargetCube"' not in ENV_SOURCE.split(
        "shelf_floor_contact = ContactSensorCfg(", 1
    )[1].split("target_object = RigidObjectCfg(", 1)[0]
    assert "shelf_collision = RewTerm(" in ENV_SOURCE
    assert "func=mdp.shelf_collision" in ENV_SOURCE
    assert "sensor.data.force_matrix_w" in REWARD_SOURCE
    assert "sensor.data.contact_pos_w" in REWARD_SOURCE
    assert "math_utils.quat_apply_inverse" in REWARD_SOURCE
    assert "valid & on_floor & (force > force_threshold)" in REWARD_SOURCE
    assert "shelf.data.default_root_state" not in REWARD_SOURCE
    assert "shelf.data.root_vel_w" not in REWARD_SOURCE
    assert "observations" not in REWARD_SOURCE


def test_v1_exponential_position_reward_contract():
    assert "PRE_REACH_EXP_REWARD_WEIGHT = 3.0" in ENV_SOURCE
    assert "class CubePreReachRewardsCfgV1(CubePreReachRewardsCfg)" in ENV_SOURCE
    assert _class_assignments("CubePreReachRewardsCfg") == {"shelf_collision"}
    assert _class_assignments("CubePreReachRewardsCfgV1") == {
        "end_effector_position_tracking",
        "end_effector_position_tracking_fine_grained",
    }
    assert "func=mdp.tcp_position_command_reward_exp" in ENV_SOURCE
    assert "weight=PRE_REACH_EXP_REWARD_WEIGHT" in ENV_SOURCE
    assert "end_effector_position_tracking_fine_grained = None" in ENV_SOURCE
    assert "def tcp_position_command_reward_exp(" in REWARD_SOURCE
    assert "distance = tcp_position_command_error(" in REWARD_SOURCE
    assert "return torch.exp(-10.0 * distance)" in REWARD_SOURCE
    assert "rewards: CubePreReachRewardsCfgV1" in ENV_SOURCE
    assert "UR5eGripperShelfCubePreReachV1PPORunnerCfg" in AGENT_SOURCE
    assert (
        'experiment_name = "reach_shelf_ur5e_gripper_cube_pre_reach_v1"'
        in AGENT_SOURCE
    )


def test_parent_action_reward_and_ppo_contract():
    assert "class UR5eGripperShelfCubePreReachEnvCfg(" in ENV_SOURCE
    assert "actions:" not in ENV_SOURCE
    assert "rewards: CubePreReachRewardsCfg" in ENV_SOURCE
    assert "terminations:" not in ENV_SOURCE
    assert "events:" not in ENV_SOURCE
    assert "curriculum:" not in ENV_SOURCE
    assert "UR5eGripperShelfReachPPORunnerCfg" in AGENT_SOURCE
    assert 'experiment_name = "reach_shelf_ur5e_gripper_cube_pre_reach"' in AGENT_SOURCE
