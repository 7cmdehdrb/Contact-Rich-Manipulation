"""Static contracts for the standalone shelf Cube Sweep v5 package."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1]
ROOT_PACKAGE = PACKAGE.parent


def _source(relative_path: str) -> str:
    return (PACKAGE / relative_path).read_text(encoding="utf-8")


ENV_SOURCE = _source("env_cfg.py")
INIT_SOURCE = _source("__init__.py")
AGENT_SOURCE = _source("agents/rsl_rl_ppo_cfg.py")
ACTION_SOURCE = _source("mdp/actions.py")
COMMAND_SOURCE = _source("mdp/commands.py")
REWARD_SOURCE = _source("mdp/rewards.py")
TERMINATION_SOURCE = _source("mdp/terminations.py")
OBSERVATION_SOURCE = _source("mdp/observations.py")
ROOT_INIT_SOURCE = (ROOT_PACKAGE / "__init__.py").read_text(encoding="utf-8")


def _class_node(source: str, class_name: str) -> ast.ClassDef:
    return next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _class_source(source: str, class_name: str) -> str:
    segment = ast.get_source_segment(source, _class_node(source, class_name))
    assert segment is not None
    return segment


def test_package_has_no_cross_task_python_dependency():
    for path in PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("sweep_rl"), (path, node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("sweep_rl"), (path, alias.name)


def test_registration_is_owned_by_new_package():
    assert "from . import shelf_cube_sweep_v5" in ROOT_INIT_SOURCE
    assert 'TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v5"' in INIT_SOURCE
    assert "sweep_rl.shelf_cube_sweep_v5.env_cfg:" in INIT_SOURCE
    old_init = (ROOT_PACKAGE / "shelf_cube_sweep" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "Cube-v5" not in old_init
    assert "Cube-Play-v5" not in old_init


def test_environment_breaks_old_task_inheritance_chain():
    env_class = _class_node(ENV_SOURCE, "ShelfCubeSweepV5EnvCfg")
    assert len(env_class.bases) == 1
    assert isinstance(env_class.bases[0], ast.Name)
    assert env_class.bases[0].id == "ManagerBasedRLEnvCfg"
    forbidden = (
        "shelf_cube_sweep",
        "shelf_cube_pre_reach",
        "shelf_reach",
        "UR5eGripperShelfCubeSweepEnvCfg",
    )
    assert all(value not in ENV_SOURCE for value in forbidden)


def test_observation_contract_includes_ee_pose_and_sweep_goal():
    policy = _class_source(ENV_SOURCE, "PolicyCfg")
    assigned = [
        target.id
        for node in _class_node(ENV_SOURCE, "PolicyCfg").body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    ]
    assert assigned == [
        "joint_pos",
        "joint_vel",
        "pose_command",
        "target_object_position",
        "cube_width",
        "current_ee_pose",
        "sweep_goal_position",
        "actions",
    ]
    assert "func=mdp.current_ee_pose_b" in policy
    assert "func=mdp.sweep_goal_position_b" in policy
    assert "def current_ee_pose_b(" in OBSERVATION_SOURCE
    assert "def sweep_goal_position_b(" in OBSERVATION_SOURCE
    assert "command.goal_pos_w" in OBSERVATION_SOURCE
    assert "phase" not in policy


def test_reward_and_termination_contract():
    rewards = _class_source(ENV_SOURCE, "RewardsCfg")
    for term in (
        "end_effector_position_tracking",
        "end_effector_orientation_tracking",
        "action_rate",
        "joint_vel",
        "shelf_collision",
        "pushing_target",
        "homing_after_sweep",
    ):
        assert f"{term} = RewTerm(" in rewards
    assert "REACH_REWARD_WEIGHT = 3.0" in ENV_SOURCE
    assert "PUSH_REWARD_WEIGHT = 2.0 * REACH_REWARD_WEIGHT" in ENV_SOURCE
    assert "HOME_REWARD_WEIGHT = 9.0" in ENV_SOURCE
    assert "HOME_SUCCESS_DISTANCE = 0.03" in ENV_SOURCE
    assert "HOME_GATE_SHARPNESS = 100.0" in ENV_SOURCE
    assert "ORIENTATION_REWARD_WEIGHT = 2.0" in ENV_SOURCE
    assert "func=mdp.align_ee_to_shelf" in rewards
    assert "weight=-0.03" in rewards
    assert rewards.count("weight=-0.03") == 2
    assert "SHELF_COLLISION_WEIGHT = -10.0" in ENV_SOURCE
    assert "ARM_JOINT_NAMES[:5]" in ENV_SOURCE
    assert "command.goal_pos_w[:, 1:]" in REWARD_SOURCE
    assert "target.data.root_pos_w[:, 1:]" in REWARD_SOURCE
    assert "torch.tanh(" in REWARD_SOURCE
    assert "success_latched" not in COMMAND_SOURCE + REWARD_SOURCE + ENV_SOURCE
    terminations = _class_source(ENV_SOURCE, "TerminationsCfg")
    assert terminations.count("DoneTerm(") == 5
    assert "time_out" in terminations
    for term in (
        "cube_drop",
        "cube_tilt",
        "cube_too_fast",
        "arm_joint_too_fast",
    ):
        assert f"{term} = DoneTerm(" in terminations
    assert "success" not in terminations


def test_failure_termination_contract():
    assert "CUBE_DROP_MIN_CENTER_HEIGHT = ACTIVE_SHELF_SURFACE_HEIGHT - 0.01" in ENV_SOURCE
    assert "CUBE_MAX_TILT_RADIANS = 0.9" in ENV_SOURCE
    assert "CUBE_MAX_LINEAR_SPEED = 0.3" in ENV_SOURCE
    assert "ARM_MAX_JOINT_SPEED = 1.0" in ENV_SOURCE
    assert "def cube_below_minimum_height(" in TERMINATION_SOURCE
    assert "def cube_excessive_tilt(" in TERMINATION_SOURCE
    assert "def cube_too_fast(" in TERMINATION_SOURCE
    assert "def arm_joint_too_fast(" in TERMINATION_SOURCE
    assert "math_utils.quat_apply" in TERMINATION_SOURCE
    assert "torch.acos" in TERMINATION_SOURCE
    assert "root_lin_vel_w" in TERMINATION_SOURCE
    assert "robot.data.joint_vel[:, asset_cfg.joint_ids]" in TERMINATION_SOURCE


def test_action_and_ppo_stability_contract():
    actions = _class_source(ENV_SOURCE, "ActionsCfg")
    assert "mdp.ArmJointPositionWithOpenGripperActionCfg(" in actions
    assert "preserve_order=False" in actions
    assert "gripper_open_position=0.0" in actions
    assert "gripper_action" not in actions
    assert "class ArmJointPositionWithOpenGripperAction(" in ACTION_SOURCE
    assert "self._asset.set_joint_position_target(" in ACTION_SOURCE
    assert "RateLimited" not in actions + ACTION_SOURCE
    assert "max_delta" not in actions + ACTION_SOURCE
    assert "Hysteresis" not in actions + ACTION_SOURCE
    assert AGENT_SOURCE.count("hidden_dims=[256, 128, 64]") == 2
    assert "clip_actions = None" in AGENT_SOURCE
    assert "num_steps_per_env = 36" in AGENT_SOURCE
    assert "max_iterations = 90_000" in AGENT_SOURCE
    assert "init_std=1.0" in AGENT_SOURCE
    assert 'std_type="scalar"' in AGENT_SOURCE
    assert "entropy_coef=0.005" in AGENT_SOURCE
    assert "learning_rate=1.0e-3" in AGENT_SOURCE
    assert 'schedule="adaptive"' in AGENT_SOURCE
    assert "gamma=0.98" in AGENT_SOURCE
    assert "desired_kl=0.02" in AGENT_SOURCE
    assert "MAX_EPISODE_STEPS = 500" in ENV_SOURCE
    assert "self.sim.dt = 0.01" in ENV_SOURCE
    assert "self.decimation = 2" in ENV_SOURCE


def test_homing_metrics_contract():
    assert 'self.metrics["goal_xyz_distance"]' in COMMAND_SOURCE
    assert 'self.metrics["goal_yz_distance"]' in COMMAND_SOURCE
    assert 'self.metrics["cube_y_displacement"]' in COMMAND_SOURCE
    assert '"cube_y_delta"' in COMMAND_SOURCE
    assert 'self.metrics["cube_y_velocity"]' in COMMAND_SOURCE
    assert '"normalized_progress_rate"' in COMMAND_SOURCE
    assert 'self.metrics["tcp_contact_distance"]' in COMMAND_SOURCE
    assert 'self.metrics["wrist_y_error"]' in COMMAND_SOURCE
    assert 'self.metrics["zeta_active"]' in COMMAND_SOURCE
    assert 'self.metrics["home_gate"]' in COMMAND_SOURCE
    assert 'self.metrics["home_joint_error_5"]' in COMMAND_SOURCE
    assert "home_joint_names=list(ARM_JOINT_NAMES[:5])" in ENV_SOURCE


def test_random_sweep_contact_point_contract():
    commands = _class_source(ENV_SOURCE, "CommandsCfg")
    assert "CONTACT_POINT_BEHIND_WIDTH_SCALE = 0.5" in ENV_SOURCE
    assert "CONTACT_POINT_X_OFFSET = 0.0" in ENV_SOURCE
    assert "CONTACT_POINT_Z_OFFSET = 0.0" in ENV_SOURCE
    assert "behind_width_scale=CONTACT_POINT_BEHIND_WIDTH_SCALE" in commands
    assert "x_offset=CONTACT_POINT_X_OFFSET" in commands
    assert "z_offset=CONTACT_POINT_Z_OFFSET" in commands
    for staged_name in (
        "push_approach_latched",
        "push_approach_alpha",
        "pre_reach_dwell",
        "pre_reach_completion_distance",
        "push_reach_surface_penetration",
        "push_reach_transition_time",
    ):
        assert staged_name not in COMMAND_SOURCE + REWARD_SOURCE + ENV_SOURCE
    assert "position[:, 0] += self.cfg.x_offset" in COMMAND_SOURCE
    assert "position[:, 1] -= self.cfg.cube_width" in COMMAND_SOURCE
    assert "position[:, 2] += self.cfg.z_offset" in COMMAND_SOURCE
    assert "command.reach_target_pos_w" in REWARD_SOURCE
    assert "class PushingTargetDeltaReward(ManagerTermBase)" in REWARD_SOURCE
    assert "zeta_m = (contact_distance < gate_distance)" in REWARD_SOURCE


def test_bottom_heavy_cube_and_self_collision_contract():
    assert "CUBE_COM_OFFSET = (0.0, 0.0, -0.08)" in ENV_SOURCE
    assert "articulation_props=sim_utils.ArticulationRootPropertiesCfg(" in ENV_SOURCE
    assert "enabled_self_collisions=True" in ENV_SOURCE


def test_random_sweep_reward_math_contract():
    assert "def signed_axis_alignment(" in REWARD_SOURCE
    assert "torch.sign(alignment) * alignment.square()" in REWARD_SOURCE
    assert "def align_ee_to_shelf(" in REWARD_SOURCE
    assert "distance < 0.03" in REWARD_SOURCE
    assert "2.0 * torch.exp(-5.0 * distance)" in REWARD_SOURCE
    assert "zeta_m.float() * (progress_delta + velocity_reward)" in REWARD_SOURCE
    assert "delta_y / (goal_distance * env.step_dt)" in REWARD_SOURCE
    assert "1.0 - distance / 0.18" not in REWARD_SOURCE
    assert "target.data.root_lin_vel_w[:, 1]" in REWARD_SOURCE
