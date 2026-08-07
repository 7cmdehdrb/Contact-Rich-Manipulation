"""Static contracts for the inherited shelf Cube Sweep environment."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1]
ROOT_PACKAGE = PACKAGE.parent
ENV_SOURCE = (PACKAGE / "env_cfg.py").read_text()
COMMAND_SOURCE = (PACKAGE / "mdp" / "commands.py").read_text()
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
    assert "BEHIND_WIDTH_SCALE = 1.2" in PRE_REACH_ENV_SOURCE
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
    assert "contact_distance < 0.04" in REWARD_SOURCE
    assert "wrist_y_error < 0.04" in REWARD_SOURCE
    assert "target_y_speed > 0.05" in REWARD_SOURCE
    assert "target_y_speed < 0.10" in REWARD_SOURCE
    assert "distance < 0.03" in REWARD_SOURCE
    assert "2.0 * torch.exp(-5.0 * distance)" in REWARD_SOURCE
    assert "1.0 - distance / 0.18" in REWARD_SOURCE
    assert "reach_complete" not in REWARD_SOURCE


def test_observation_and_agent_contract():
    assert "observations:" not in ENV_SOURCE
    assert "UR5eGripperShelfCubePreReachPPORunnerCfg" in AGENT_SOURCE
    assert 'experiment_name = "sweep_shelf_ur5e_gripper_cube"' in AGENT_SOURCE
