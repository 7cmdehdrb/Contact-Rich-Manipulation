"""Static contracts for the inherited shelf Cube Sweep environment."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).parents[1]
ROOT_PACKAGE = PACKAGE.parent
ENV_SOURCE = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
COMMAND_SOURCE = (PACKAGE / "mdp" / "commands.py").read_text(encoding="utf-8")
ACTION_SOURCE = (PACKAGE / "mdp" / "actions.py").read_text(encoding="utf-8")
ASSET_SOURCE = (PACKAGE / "mdp" / "assets.py").read_text(encoding="utf-8")
REWARD_SOURCE = (PACKAGE / "mdp" / "rewards.py").read_text(encoding="utf-8")
OBSERVATION_SOURCE = (PACKAGE / "mdp" / "observations.py").read_text(
    encoding="utf-8"
)
SELF_COLLISION_SOURCE = (PACKAGE / "mdp" / "self_collision.py").read_text(
    encoding="utf-8"
)
INIT_SOURCE = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
AGENT_SOURCE = (PACKAGE / "agents" / "rsl_rl_ppo_cfg.py").read_text(
    encoding="utf-8"
)
ROOT_INIT_SOURCE = (ROOT_PACKAGE / "__init__.py").read_text(encoding="utf-8")
PRE_REACH_ENV_SOURCE = (
    ROOT_PACKAGE / "shelf_cube_pre_reach" / "env_cfg.py"
).read_text(encoding="utf-8")
PRE_REACH_COMMAND_SOURCE = (
    ROOT_PACKAGE / "shelf_cube_pre_reach" / "mdp" / "commands.py"
).read_text(encoding="utf-8")
PRE_REACH_REWARD_SOURCE = (
    ROOT_PACKAGE / "shelf_cube_pre_reach" / "mdp" / "rewards.py"
).read_text(encoding="utf-8")
REACH_ENV_SOURCE = (ROOT_PACKAGE / "shelf_reach" / "env_cfg.py").read_text(
    encoding="utf-8"
)


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
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v2" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v2" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v3" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v3" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v4" in INIT_SOURCE
    assert "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v4" in INIT_SOURCE
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
    assert _has_base(
        ENV_SOURCE,
        "UR5eGripperShelfCubeSweepEnvCfgV2",
        "UR5eGripperShelfCubeSweepEnvCfgV1",
    )
    assert _has_base(
        ENV_SOURCE,
        "CubeSweepRewardsCfgV2",
        "CubeSweepRewardsCfgV1",
    )
    assert _has_base(
        ENV_SOURCE,
        "UR5eGripperShelfCubeSweepEnvCfgV3",
        "UR5eGripperShelfCubeSweepEnvCfgV2",
    )
    assert _has_base(
        ENV_SOURCE,
        "UR5eGripperShelfCubeSweepEnvCfgV4",
        "UR5eGripperShelfCubeSweepEnvCfgV3",
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
    assert "end_effector_position_tracking_fine_grained = RewTerm(" in REACH_ENV_SOURCE
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
    assert "velocity_reward_min: float = 0.05" in REWARD_SOURCE
    assert "velocity_penalty_min: float = 0.10" in REWARD_SOURCE
    assert "target_speed > velocity_reward_min" in REWARD_SOURCE
    assert "target_speed < velocity_penalty_min" in REWARD_SOURCE
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
    assert "UR5eGripperShelfCubeSweepV2PPORunnerCfg" in AGENT_SOURCE
    assert 'experiment_name = "sweep_shelf_ur5e_gripper_cube_v2"' in AGENT_SOURCE


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


def test_v2_episode_geometry_and_speed_contract():
    assert "MAX_EPISODE_STEPS_V2 = 360" in ENV_SOURCE
    assert "math.nextafter(episode_duration, 0.0)" in ENV_SOURCE
    assert "CUBE_WIDTH_V2 = CUBE_WIDTH" in ENV_SOURCE
    assert "CUBE_DEPTH_V2 = CUBE_DEPTH" in ENV_SOURCE
    assert "size=(CUBE_WIDTH_V2, CUBE_DEPTH_V2, CUBE_HEIGHT)" in ENV_SOURCE
    assert "center_of_mass=CUBE_COM_OFFSET_V1" in ENV_SOURCE
    assert "ARM_TARGET_MAX_DELTA_V2 = 0.03" in ENV_SOURCE
    assert "WRIST_TARGET_MAX_DELTA_V2 = 0.04" in ENV_SOURCE
    assert "ARM_VELOCITY_LIMIT_V2 = 1.5" in ENV_SOURCE
    assert "WRIST_VELOCITY_LIMIT_V2 = 2.0" in ENV_SOURCE


def test_v2_slow_push_reward_contract():
    assert "PUSH_VELOCITY_REWARD_MIN_V2 = 0.03" in ENV_SOURCE
    assert "PUSH_VELOCITY_PENALTY_MIN_V2 = 0.06" in ENV_SOURCE
    assert '"velocity_reward_min": PUSH_VELOCITY_REWARD_MIN_V2' in ENV_SOURCE
    assert '"velocity_penalty_min": PUSH_VELOCITY_PENALTY_MIN_V2' in ENV_SOURCE
    assert "func=mdp.pushing_target" in ENV_SOURCE
    assert "PUSH_MAX_ABS_Y_VELOCITY_V2" not in ENV_SOURCE
    assert "PushingTargetWithSpeedLatch" not in REWARD_SOURCE
    assert "speed_limit_exceeded" not in REWARD_SOURCE
    assert '"max_abs_y_velocity"' not in ENV_SOURCE
    assert "target_speed = torch.abs(target_velocity)" in REWARD_SOURCE
    assert "target.data.root_lin_vel_w[:, 1]" in REWARD_SOURCE
    assert 'velocity_measurement: str = "com_y"' in REWARD_SOURCE
    assert "slow_push_speed = RewTerm(" not in ENV_SOURCE
    assert "def slow_push_speed(" not in REWARD_SOURCE
    assert '"include_velocity_shaping": False' not in ENV_SOURCE
    assert "V2_DENSE_REACH_DISTANCE_WEIGHT" not in ENV_SOURCE
    assert "end_effector_position_distance = RewTerm(" not in ENV_SOURCE


def test_v2_cube_vy_metric_support_is_retained_but_not_enabled():
    assert "record_cube_vy_metric: bool = False" in COMMAND_SOURCE
    assert 'self.metrics["cube_vy"]' in COMMAND_SOURCE
    assert "self.target.data.root_lin_vel_w[:, 1]" in COMMAND_SOURCE
    assert "record_cube_vy_metric=True" not in ENV_SOURCE
    v2_commands = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "CubeSweepCommandsCfgV2"),
    )
    assert v2_commands is not None
    assert "record_cube_vy_metric" not in v2_commands
    assert "record_self_collision_metrics" not in v2_commands
    assert "cube_vy = ObsTerm(" not in ENV_SOURCE


def test_v2_ppo_matches_the_pre_stabilization_runner_contract():
    v2_agent = ast.get_source_segment(
        AGENT_SOURCE,
        _class_node(AGENT_SOURCE, "UR5eGripperShelfCubeSweepV2PPORunnerCfg"),
    )
    assert v2_agent is not None
    assert "clip_actions" not in v2_agent
    assert "actor =" not in v2_agent
    assert "algorithm =" not in v2_agent
    assert _has_base(
        AGENT_SOURCE,
        "UR5eGripperShelfCubeSweepV2PPORunnerCfg",
        "UR5eGripperShelfCubeSweepV1PPORunnerCfg",
    )


def test_v3_ppo_learning_rate_policy():
    v3_agent = ast.get_source_segment(
        AGENT_SOURCE,
        _class_node(AGENT_SOURCE, "UR5eGripperShelfCubeSweepV3PPORunnerCfg"),
    )
    assert v3_agent is not None
    assert _has_base(
        AGENT_SOURCE,
        "UR5eGripperShelfCubeSweepV3PPORunnerCfg",
        "UR5eGripperShelfCubeSweepV2PPORunnerCfg",
    )
    assert 'experiment_name = "sweep_shelf_ur5e_gripper_cube_v3"' in v3_agent
    assert "deepcopy(UR5eGripperShelfCubeSweepV2PPORunnerCfg().algorithm)" in v3_agent
    assert "algorithm.learning_rate = 3.0e-4" in v3_agent
    assert 'algorithm.schedule = "fixed"' in v3_agent
    assert "entropy_coef" not in v3_agent
    assert "num_learning_epochs" not in v3_agent
    assert "clip_actions" not in v3_agent


def test_v3_uses_actor_origin_planar_speed_for_push_reward():
    v3_rewards = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "CubeSweepRewardsCfgV3"),
    )
    v3_env = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "UR5eGripperShelfCubeSweepEnvCfgV3"),
    )
    assert v3_rewards is not None
    assert v3_env is not None
    assert _has_base(
        ENV_SOURCE,
        "CubeSweepRewardsCfgV3",
        "CubeSweepRewardsCfgV2",
    )
    assert '"velocity_measurement": "root_link_planar"' in v3_rewards
    assert "rewards: CubeSweepRewardsCfgV3 = CubeSweepRewardsCfgV3()" in v3_env
    assert "target.data.root_link_lin_vel_w[:, :2]" in REWARD_SOURCE
    assert "torch.linalg.vector_norm(" in REWARD_SOURCE


def test_v4_current_ee_pose_observation_contract():
    v4_observations = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "CubeSweepObservationsCfgV4"),
    )
    assert v4_observations is not None
    assert _has_base(
        ENV_SOURCE,
        "CubeSweepObservationsCfgV4",
        "CubeSweepObservationsCfgV2",
    )
    assert "current_ee_pose = ObsTerm(" in v4_observations
    assert "func=mdp.end_effector_pose_b" in v4_observations
    assert 'SceneEntityCfg("ee_frame")' in v4_observations
    assert "def end_effector_pose_b(" in OBSERVATION_SOURCE
    assert "math_utils.subtract_frame_transforms(" in OBSERVATION_SOURCE
    assert "math_utils.quat_unique(quaternion_b)" in OBSERVATION_SOURCE
    assert "torch.cat((position_b," in OBSERVATION_SOURCE


def test_v4_directional_speed_and_push_geometry_contract():
    v4_rewards = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "CubeSweepRewardsCfgV4"),
    )
    assert v4_rewards is not None
    assert "PUSH_BEHIND_WIDTH_SCALE_V4 = 1.0" in ENV_SOURCE
    assert "SWEEP_GATE_DISTANCE_V4 = 0.04" in ENV_SOURCE
    assert "PUSH_VELOCITY_REWARD_MIN_V4 = 0.05" in ENV_SOURCE
    assert "PUSH_VELOCITY_PENALTY_MIN_V4 = 0.10" in ENV_SOURCE
    assert '"behind_width_scale": PUSH_BEHIND_WIDTH_SCALE_V4' in v4_rewards
    assert '"gate_distance": SWEEP_GATE_DISTANCE_V4' in v4_rewards
    assert '"velocity_reward_min": PUSH_VELOCITY_REWARD_MIN_V4' in v4_rewards
    assert '"velocity_penalty_min": PUSH_VELOCITY_PENALTY_MIN_V4' in v4_rewards
    assert '"velocity_measurement": "com_y"' in v4_rewards
    assert '"max_tilt_radians"' not in v4_rewards


def test_v4_original_action_space_contract():
    v4_actions = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "CubeSweepActionsCfgV4"),
    )
    assert v4_actions is not None
    assert "base_mdp.JointPositionActionCfg(" in v4_actions
    assert "scale=0.5" in v4_actions
    assert "use_default_offset=True" in v4_actions
    assert "base_mdp.BinaryJointPositionActionCfg(" in v4_actions
    assert "open_command_expr={" in v4_actions
    assert v4_actions.count(": 0.0") >= 8
    assert '"finger_joint": 0.5' in v4_actions
    assert '"left_inner_finger_joint": -0.5' in v4_actions
    assert '"right_inner_finger_joint": 0.5' in v4_actions


def test_v4_actor_network_contract():
    v4_agent = ast.get_source_segment(
        AGENT_SOURCE,
        _class_node(AGENT_SOURCE, "UR5eGripperShelfCubeSweepV4PPORunnerCfg"),
    )
    assert v4_agent is not None
    assert _has_base(
        AGENT_SOURCE,
        "UR5eGripperShelfCubeSweepV4PPORunnerCfg",
        "UR5eGripperShelfCubeSweepV3PPORunnerCfg",
    )
    assert 'experiment_name = "sweep_shelf_ur5e_gripper_cube_v4"' in v4_agent
    assert "deepcopy(UR5eGripperShelfCubeSweepV3PPORunnerCfg().actor)" in v4_agent
    assert "actor.hidden_dims = [256, 128, 64]" in v4_agent


def test_v2_kinematic_shelf_and_collision_penalty_contract():
    assert "kinematic_enabled=True" in ENV_SOURCE
    assert "disable_gravity=True" in ENV_SOURCE
    assert "SHELF_COLLISION_WEIGHT_V2 = 2.0 * SHELF_COLLISION_WEIGHT" in ENV_SOURCE
    assert "SHELF_FLOOR_SURFACE_HEIGHTS_V2 = (0.70, 1.05, 1.50)" in ENV_SOURCE
    assert "weight=SHELF_COLLISION_WEIGHT_V2" in ENV_SOURCE
    assert '"surface_heights": SHELF_FLOOR_SURFACE_HEIGHTS_V2' in ENV_SOURCE
    assert "func=cube_pre_reach_mdp.shelf_collision" in ENV_SOURCE
    assert "shelf_tipped_over" not in ENV_SOURCE


def test_v2_self_collision_contract():
    assert "enabled_self_collisions=True" in ENV_SOURCE
    assert "robot_cfg.spawn.activate_contact_sensors = True" in ENV_SOURCE
    assert "SELF_COLLISION_SENSOR_NAMES_V2 = (" in ENV_SOURCE
    assert "SELF_COLLISION_EXCLUDED_PAIRS_V2 = (" in ENV_SOURCE
    assert "class CubeSweepTerminationsCfgV2(TerminationsCfg):" in ENV_SOURCE
    assert "func=mdp.SelfCollisionTermination" in ENV_SOURCE
    assert "time_out=False" in ENV_SOURCE
    assert "SELF_COLLISION_SOFT_FORCE_THRESHOLD_V2 = 2.0" in ENV_SOURCE
    assert "SELF_COLLISION_PERSISTENT_FORCE_THRESHOLD_V2 = 5.0" in ENV_SOURCE
    assert "SELF_COLLISION_HARD_FORCE_THRESHOLD_V2 = 20.0" in ENV_SOURCE
    assert "SELF_COLLISION_PERSISTENCE_STEPS_V2 = 2" in ENV_SOURCE
    assert "self_collision_force = RewTerm(" in ENV_SOURCE
    assert "self_collision_terminal = RewTerm(" in ENV_SOURCE
    assert "SELF_COLLISION_TERMINAL_BASE_COST_V2 = 10.0" in ENV_SOURCE
    assert "SELF_COLLISION_TERMINAL_EARLY_COST_V2 = 30.0" in ENV_SOURCE
    assert "class CubeSweepEventsCfgV2(CubeSweepEventsCfgV1):" in ENV_SOURCE
    assert '"position_range": (0.95, 1.05)' in ENV_SOURCE
    v2_env = ast.get_source_segment(
        ENV_SOURCE,
        _class_node(ENV_SOURCE, "UR5eGripperShelfCubeSweepEnvCfgV2"),
    )
    assert v2_env is not None
    assert "ShelfCubeSweepSelfCollisionSceneCfgV2" not in v2_env
    assert "CubeSweepSelfCollisionRewardsCfgV2" not in v2_env
    assert "CubeSweepTerminationsCfgV2" not in v2_env
    assert "CubeSweepEventsCfgV2" not in v2_env
    assert "events: CubeSweepEventsCfgV1" in v2_env
    assert "class SelfCollisionTermination(ManagerTermBase):" in SELF_COLLISION_SOURCE
    assert "filter_id in range(sensor_id + 1" in SELF_COLLISION_SOURCE
    assert "consecutive_contact_steps >= persistence_steps" in SELF_COLLISION_SOURCE
    assert "return terminated * cost / env.step_dt" in SELF_COLLISION_SOURCE
    assert 'command.metrics["self_collision_force_max"]' in SELF_COLLISION_SOURCE
    assert (
        'command.metrics["self_collision_termination_rate"]'
        in SELF_COLLISION_SOURCE
    )
