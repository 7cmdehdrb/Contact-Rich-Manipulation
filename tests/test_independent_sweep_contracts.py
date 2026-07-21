"""Static contracts for the Isaac-dependent standalone sweep environment."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sweep_rl"
    / "sweep_rl"
    / "osc_sweep_independent"
)


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _assigned_names(class_node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for node in class_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            names.append(node.targets[0].id)
    return names


def test_environment_is_direct_and_rewards_are_exactly_approved_terms() -> None:
    classes = _classes(PACKAGE / "env_cfg.py")
    environment = classes["UR5eOscSweepIndependentEnvCfg"]
    assert len(environment.bases) == 1
    assert isinstance(environment.bases[0], ast.Name)
    assert environment.bases[0].id == "ManagerBasedRLEnvCfg"
    assert _assigned_names(classes["RewardsCfg"]) == [
        "reaching",
        "contact",
        "push",
        "home_return",
    ]


def test_policy_observation_contract_is_56_dimensions() -> None:
    expected_dimensions = {
        "joint_pos": 6,
        "joint_vel": 6,
        "joint_effort": 6,
        "eef_pose": 6,
        "ft_sensor": 6,
        "contact_point": 3,
        "initial_target_pose": 6,
        "desired_motion": 4,
        "task_phase": 1,
        "last_action": 12,
    }
    source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    for term_name in expected_dimensions:
        assert f"{term_name} = ObsTerm(" in source
    assert sum(expected_dimensions.values()) == 56
    assert "current_target_pose" not in source
    assert "object_linear_velocity" not in source


def test_action_and_randomization_ranges_match_the_task_contract() -> None:
    environment_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    action_source = (PACKAGE / "mdp" / "actions.py").read_text(encoding="utf-8")
    assert "distance_range=(0.12, 0.35)" in environment_source
    assert "target_speed_range=(0.04, 0.12)" in environment_source
    assert '"size_range": (0.04, 0.08)' in environment_source
    assert '"mass_distribution_params": (0.25, 2.0)' in environment_source
    assert "Expected a 12-D OSC action" in action_source
    assert "self._gripper_open_targets" in action_source


def test_three_task_phases_gate_their_own_rewards_and_transitions() -> None:
    common_source = (PACKAGE / "mdp" / "common.py").read_text(encoding="utf-8")
    command_source = (PACKAGE / "mdp" / "commands.py").read_text(encoding="utf-8")
    reward_source = (PACKAGE / "mdp" / "rewards.py").read_text(encoding="utf-8")
    assert "PHASE_REACH = 0" in common_source
    assert "PHASE_SWEEP = 1" in common_source
    assert "PHASE_HOME = 2" in common_source
    assert "self.task_phase[ids] = PHASE_REACH" in command_source
    assert (
        "entering_sweep = (self.task_phase == PHASE_REACH) & target_contact"
        in command_source
    )
    assert "self.task_phase == PHASE_SWEEP" in command_source
    assert "torch.full_like(self.task_phase, PHASE_HOME)" in command_source
    assert "command.task_phase == PHASE_REACH" in reward_source
    assert reward_source.count("task_phase == PHASE_SWEEP") == 2
    assert "command.task_phase == PHASE_HOME" in reward_source


def test_goal_sampling_uses_measured_shelf_board_bounds() -> None:
    environment_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    command_source = (PACKAGE / "mdp" / "commands.py").read_text(encoding="utf-8")
    assert "SHELF_SURFACE_HEIGHT = 1.05" in environment_source
    assert "SHELF_WORKSPACE_X_RANGE = (0.50, 0.90)" in environment_source
    assert "SHELF_WORKSPACE_Y_RANGE = (-0.50, 0.50)" in environment_source
    assert "workspace_boundary_margin=0.015" in environment_source
    assert "margin = self.cfg.workspace_boundary_margin + half_size" in command_source
    assert "direction_b, maximum = self._sample_feasible_direction" in command_source


def test_scene_uses_shelf_pose_and_keeps_sensor_equipped_robot() -> None:
    source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    assert (
        '"omniverse://192.168.0.13/Library/Shelf/Arena/Collected_speedrack_shape/speedrack_shape.usd"'
        in source
    )
    assert 'prim_path="{ENV_REGEX_NS}/Shelf"' in source
    assert "mass=100.0" not in source
    assert "robot_cfg = make_ur5e_robotiq_ft_cfg()" in source
    assert "robot_cfg.init_state.pos = (0.0, 0.0, 0.79505)" in source
    assert "robot_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)" in source
    assert '"shoulder_lift_joint": -2.2' in source
    assert '"elbow_joint": 2.2' in source
    assert '"wrist_2_joint": 1.57' in source
    assert '"wrist_3_joint": 0.785' in source

    scene = _classes(PACKAGE / "env_cfg.py")["IndependentSweepSceneCfg"]
    scene_names = _assigned_names(scene)
    assert "shelf" in scene_names
    assert "table" not in scene_names
    assert {"left_contact", "right_contact", "base_contact"} <= set(scene_names)
    assert "target_robot_contact" not in scene_names


def test_robot_contact_filters_are_one_body_to_many_and_terminate_on_shelf() -> None:
    environment_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    asset_source = (PACKAGE.parent / "osc_sweep" / "assets.py").read_text(
        encoding="utf-8"
    )
    assert '"{ENV_REGEX_NS}/Robot/.*"' not in environment_source
    assert '"{ENV_REGEX_NS}/TargetCube"' in environment_source
    assert '"{ENV_REGEX_NS}/Shelf/rack"' in environment_source
    assert environment_source.count("make_robot_body_contact_sensor(") == 19
    assert "func=mdp.robot_shelf_collision" in environment_source
    assert '"shelf_filter_index": SHELF_CONTACT_FILTER_INDEX' in environment_source
    assert '"force_threshold": 0.1' in environment_source
    assert "activate_contact_sensors=cfg.activate_contact_sensors" in asset_source
    assert "_deinstance_geometry(stage, prim_path)" in asset_source
    assert "enabled_self_collisions = True" in environment_source
    assert "func=mdp.robot_self_collision" in environment_source
    assert (
        '"self_filter_start_index": SELF_CONTACT_FILTER_START_INDEX'
        in environment_source
    )
    assert '"excluded_pairs": SELF_COLLISION_EXCLUDED_PAIRS' in environment_source
    termination_source = (PACKAGE / "mdp" / "terminations.py").read_text(
        encoding="utf-8"
    )
    assert "self_forces > force_threshold" in termination_source
    assert "ignored_filter_indices" in termination_source


def test_observation_noise_configuration_is_hydra_serializable() -> None:
    environment_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    observation_source = (PACKAGE / "mdp" / "observations.py").read_text(
        encoding="utf-8"
    )
    assert "torch.tensor(" not in environment_source
    assert environment_source.count("noise=mdp.VectorUniformNoiseCfg(") == 2
    assert "n_min: list[float] = MISSING" in observation_source
    assert "n_max: list[float] = MISSING" in observation_source
    assert "torch.as_tensor(cfg.n_min" in observation_source
    assert "torch.as_tensor(cfg.n_max" in observation_source


def test_friction_randomization_and_goal_visualization_are_enabled() -> None:
    environment_source = (PACKAGE / "env_cfg.py").read_text(encoding="utf-8")
    command_source = (PACKAGE / "mdp" / "commands.py").read_text(
        encoding="utf-8"
    )
    assert "randomize_target_friction = EventTerm(" in environment_source
    assert "randomize_shelf_friction = EventTerm(" in environment_source
    assert environment_source.count("func=mdp.randomize_rigid_body_material") == 2
    assert environment_source.count('"static_friction_range": (0.40, 1.10)') == 2
    assert environment_source.count('"dynamic_friction_range": (0.25, 0.90)') == 2
    assert environment_source.count('"make_consistent": True') == 2
    assert "debug_vis=True" in environment_source
    event_source = (PACKAGE / "mdp" / "events.py").read_text(encoding="utf-8")
    assert "create_position_visualizers = EventTerm(" in environment_source
    assert "func=mdp.create_sweep_position_visualizers" in environment_source
    assert event_source.count("VisualizationMarkers(") == 2
    assert event_source.count("markers={") == 2
    assert "mode=\"prestartup\"" in environment_source
    assert "marker_indices=" not in command_source
    assert command_source.count(".visualize(") == 2
    assert "visualization_height_offset: float = 0.10" in command_source


def test_package_readme_documents_observation_action_reward_and_randomization() -> None:
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "## 2. Observation" in readme
    assert "**합계** | **56**" in readme
    assert "## 3. Action" in readme
    assert "12-D policy action" in readme
    assert "## 4. Reward" in readme
    assert all(
        term in readme for term in ("Reaching", "Contact", "Push", "HomeReturn")
    )
    assert "## 5. Domain Randomization" in readme
    assert "`prestartup`" in readme
    assert "`startup`" in readme
    assert "`reset`" in readme
    assert "Object–Shelf friction" in readme
