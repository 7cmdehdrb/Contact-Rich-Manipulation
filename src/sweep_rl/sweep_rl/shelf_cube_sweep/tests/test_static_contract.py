"""Static contracts for the inherited shelf Cube Sweep environment."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1]
ROOT_PACKAGE = PACKAGE.parent
ENV_SOURCE = (PACKAGE / "env_cfg.py").read_text()
COMMAND_SOURCE = (PACKAGE / "mdp" / "commands.py").read_text()
ACTION_SOURCE = (PACKAGE / "mdp" / "actions.py").read_text()
ASSET_SOURCE = (PACKAGE / "mdp" / "assets.py").read_text()
REWARD_SOURCE = (PACKAGE / "mdp" / "rewards.py").read_text()
INIT_SOURCE = (PACKAGE / "__init__.py").read_text()
AGENT_SOURCE = (PACKAGE / "agents" / "rsl_rl_ppo_cfg.py").read_text()
ROOT_INIT_SOURCE = (ROOT_PACKAGE / "__init__.py").read_text()
PRE_REACH_ENV_SOURCE = (
    ROOT_PACKAGE / "shelf_cube_pre_reach" / "env_cfg.py"
).read_text()
PRE_REACH_COMMAND_SOURCE = (
    ROOT_PACKAGE / "shelf_cube_pre_reach" / "mdp" / "commands.py"
).read_text()
PRE_REACH_REWARD_SOURCE = (
    ROOT_PACKAGE / "shelf_cube_pre_reach" / "mdp" / "rewards.py"
).read_text()
REACH_ENV_SOURCE = (ROOT_PACKAGE / "shelf_reach" / "env_cfg.py").read_text()


def _class_node(source: str, class_name: str) -> ast.ClassDef:
    module = ast.parse(source)
    return next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _has_base(source: str, class_name: str, base_name: str) -> bool:
    return any(
        isinstance(base, ast.Name) and base.id == base_name
        for base in _class_node(source, class_name).bases
    )


def test_task_registration_and_environment_inheritance():
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v1" in INIT_SOURCE
    assert "from . import shelf_cube_sweep" in ROOT_INIT_SOURCE
    assert _has_base(
        ENV_SOURCE,
        "UR5eGripperShelfCubeSweepEnvCfg",
        "UR5eGripperShelfCubePreReachEnvCfg",
    )
    assert _has_base(
        ENV_SOURCE,
        "ShelfCubeSweepSceneCfg",
        "ShelfCubePreReachSceneCfg",
    )
    assert _has_base(
        ENV_SOURCE,
        "CubeSweepCommandsCfg",
        "CubePreReachCommandsCfg",
    )
    assert _has_base(
        ENV_SOURCE,
        "CubeSweepRewardsCfg",
        "CubePreReachRewardsCfg",
    )
    assert _has_base(
        ENV_SOURCE,
        "UR5eGripperShelfCubeSweepEnvCfgV1",
        "UR5eGripperShelfCubePreReachEnvCfgV1",
    )
    assert _has_base(
        ENV_SOURCE,
        "CubeSweepRewardsCfgV1",
        "CubePreReachRewardsCfgV1",
    )


def test_pre_reach_package_contains_no_sweep_logic():
    combined = PRE_REACH_ENV_SOURCE + PRE_REACH_COMMAND_SOURCE
    assert "PUSH_GOAL_OFFSET" not in combined
    assert "PUSH_REWARD_WEIGHT" not in combined
    assert "wrist_frame" not in combined
    assert "goal_pos_w" not in combined
    assert "pushing_target" not in PRE_REACH_ENV_SOURCE
    assert "pushing_target" not in PRE_REACH_REWARD_SOURCE
    assert "resampling_time_range=(4.0, 4.0)" in PRE_REACH_ENV_SOURCE


def test_inherited_reach_orientation_and_collision_contract():
    assert "end_effector_position_tracking = RewTerm(" in REACH_ENV_SOURCE
    assert (
        "end_effector_position_tracking_fine_grained = RewTerm("
        in REACH_ENV_SOURCE
    )
    assert "end_effector_orientation_tracking = RewTerm(" in REACH_ENV_SOURCE
    assert "shelf_collision = RewTerm(" in PRE_REACH_ENV_SOURCE
    assert "pushing_target = RewTerm(" in ENV_SOURCE
    assert "end_effector_position_tracking = None" not in ENV_SOURCE
    assert "end_effector_position_tracking_fine_grained = None" not in ENV_SOURCE


def test_moving_reach_point_and_fixed_sweep_goal_contract():
    assert "self.target.data.root_pos_w[env_ids]" in PRE_REACH_COMMAND_SOURCE
    assert "def _update_command(self)" in PRE_REACH_COMMAND_SOURCE
    assert "BEHIND_WIDTH_SCALE = 1.1" in PRE_REACH_ENV_SOURCE
    assert "PRE_REACH_Z_OFFSET = 0.03" in PRE_REACH_ENV_SOURCE
    assert "PRE_REACH_X_OFFSET = 0.0" in ENV_SOURCE
    assert "PUSH_GOAL_OFFSET = (0.0, 0.18, 0.0)" in ENV_SOURCE
    assert _has_base(
        COMMAND_SOURCE,
        "CubeSweepPoseCommand",
        "CubePreReachPoseCommand",
    )
    assert "self.goal_pos_w = torch.zeros" in COMMAND_SOURCE
    assert "self.target.data.root_pos_w[ids] + goal_offset" in COMMAND_SOURCE
    assert "resampling_time_range=(1.0e9, 1.0e9)" in ENV_SOURCE


def test_requested_push_reward_contract():
    assert "PUSH_REWARD_WEIGHT = 6.0" in ENV_SOURCE
    assert "weight=PUSH_REWARD_WEIGHT" in ENV_SOURCE
    assert "SWEEP_GATE_DISTANCE = 0.08" in ENV_SOURCE
    assert "SWEEP_GATE_DISTANCE < CUBE_WIDTH * BEHIND_WIDTH_SCALE" in ENV_SOURCE
    assert '"gate_distance": SWEEP_GATE_DISTANCE' in ENV_SOURCE
    assert "contact_distance < gate_distance" in REWARD_SOURCE
    assert "wrist_y_error < gate_distance" in REWARD_SOURCE
    assert "target_y_speed > 0.05" in REWARD_SOURCE
    assert "target_y_speed < 0.10" in REWARD_SOURCE
    assert "distance < 0.03" in REWARD_SOURCE
    assert "2.0 * torch.exp(-5.0 * distance)" in REWARD_SOURCE
    assert "1.0 - distance / 0.18" in REWARD_SOURCE
    assert "reach_complete" not in REWARD_SOURCE


def test_v1_reward_and_moving_offset_contract():
    assert "PUSH_REWARD_WEIGHT_V1 = 2.0 * PUSH_REWARD_WEIGHT" in ENV_SOURCE
    assert "weight=PUSH_REWARD_WEIGHT_V1" in ENV_SOURCE
    assert "rewards: CubeSweepRewardsCfgV1" in ENV_SOURCE
    assert "commands: CubeSweepCommandsCfgV1" in ENV_SOURCE
    assert "self.target.data.root_pos_w[env_ids]" in PRE_REACH_COMMAND_SOURCE
    assert "def _update_command(self)" in PRE_REACH_COMMAND_SOURCE
    assert "class CubePreReachRewardsCfgV1(" in PRE_REACH_ENV_SOURCE
    assert "func=mdp.tcp_position_command_reward_exp" in PRE_REACH_ENV_SOURCE
    assert "weight=3.0" in PRE_REACH_ENV_SOURCE
    assert "weight=-0.7" in PRE_REACH_ENV_SOURCE
    assert "weight=-1.5" in ENV_SOURCE
    assert "weight=-0.03" in PRE_REACH_ENV_SOURCE
    assert "end_effector_position_tracking_fine_grained = None" in PRE_REACH_ENV_SOURCE


def test_observation_and_agent_contract():
    assert "class CubeSweepObservationsCfgV1(" in ENV_SOURCE
    assert "target_object_angular_velocity" not in ENV_SOURCE
    assert "root_ang_vel_w" not in ENV_SOURCE
    assert 'params={"width": CUBE_WIDTH_V1}' in ENV_SOURCE
    assert "target.data.root_quat_w" in REWARD_SOURCE
    assert "up_axis_w" in REWARD_SOURCE
    assert "UR5eGripperShelfCubePreReachPPORunnerCfg" in AGENT_SOURCE
    assert 'experiment_name = "sweep_shelf_ur5e_gripper_cube"' in AGENT_SOURCE
    assert "UR5eGripperShelfCubePreReachV1PPORunnerCfg" in AGENT_SOURCE
    assert 'experiment_name = "sweep_shelf_ur5e_gripper_cube_v1"' in AGENT_SOURCE


def test_v1_full_range_rate_limited_action_contract():
    assert "RateLimitedJointPositionToLimitsActionCfg" in ENV_SOURCE
    assert "rescale_to_limits=True" in ENV_SOURCE
    assert "ARM_TARGET_MAX_DELTA_V1 = 0.05" in ENV_SOURCE
    assert "WRIST_TARGET_MAX_DELTA_V1 = 0.07" in ENV_SOURCE
    assert "requested_target - previous_target" in ACTION_SOURCE
    assert "min=-self._max_delta" in ACTION_SOURCE
    assert "max=self._max_delta" in ACTION_SOURCE
    assert "self._processed_actions[env_ids]" in ACTION_SOURCE


def test_v1_upright_reward_and_bottom_heavy_cube_contract():
    assert "CUBE_WIDTH_V1 = 0.11" in ENV_SOURCE
    assert "CUBE_DEPTH_V1 = 0.11" in ENV_SOURCE
    assert "size=(CUBE_WIDTH_V1, CUBE_DEPTH_V1, CUBE_HEIGHT)" in ENV_SOURCE
    assert "CUBE_COM_OFFSET_V1 = (0.0, 0.0, -0.05)" in ENV_SOURCE
    assert "CUBE_DIAGONAL_INERTIA_V1 = (0.008, 0.008, 0.003)" in ENV_SOURCE
    assert "CreateCenterOfMassAttr" in ASSET_SOURCE
    assert "CreateDiagonalInertiaAttr" in ASSET_SOURCE
    assert "CUBE_MAX_TILT_RADIANS_V1 = math.radians(10.0)" in ENV_SOURCE
    assert '"max_tilt_radians": CUBE_MAX_TILT_RADIANS_V1' in ENV_SOURCE
    assert "def cube_upright_quality(" in REWARD_SOURCE
    assert "reward = reward * cube_upright_quality(" in REWARD_SOURCE


def test_v1_vertical_wrist_reset_and_curriculum_contract():
    assert "VERTICAL_WRIST_3_POSITION_V1 = math.pi / 2.0" in ENV_SOURCE
    assert 'joint_pos["wrist_3_joint"]' in ENV_SOURCE
    assert "joint_names=list(ARM_JOINT_NAMES[:3])" in ENV_SOURCE
    v1_env = _class_node(ENV_SOURCE, "UR5eGripperShelfCubeSweepEnvCfgV1")
    assigned_names = {
        target.id
        for node in v1_env.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "curriculum" in assigned_names
