"""Static contracts for the inherited detailed-reward sweep environment."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "sweep_rl" / "sweep_rl" / "osc_sweep_independent"


def _class(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _assigned_names(class_node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for node in class_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            names.append(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_detailed_task_is_registered_and_inherits_independent_environment() -> None:
    registration = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0" in registration
    assert "UR5eOscSweepIndependentDetailedEnvCfg" in registration

    environment = _class(
        PACKAGE / "env_cfg_detailed.py",
        "UR5eOscSweepIndependentDetailedEnvCfg",
    )
    assert len(environment.bases) == 1
    assert isinstance(environment.bases[0], ast.Name)
    assert environment.bases[0].id == "UR5eOscSweepIndependentEnvCfg"
    assert _assigned_names(environment) == ["rewards"]


def test_rewards_are_decomposed_by_phase_and_include_safety_regularization() -> None:
    rewards = _assigned_names(
        _class(PACKAGE / "env_cfg_detailed.py", "DetailedRewardsCfg")
    )
    assert rewards == [
        "reach_pose_tracking",
        "reach_pose_error",
        "sweep_contact",
        "sweep_velocity_tracking",
        "sweep_forward_progress",
        "sweep_endpoint_error",
        "sweep_lateral_error",
        "sweep_overshoot",
        "sweep_stopped_at_goal",
        "home_joint_pose",
        "home_joint_error",
        "home_clearance",
        "post_goal_contact",
        "goal_hold_error",
        "post_goal_object_speed",
        "post_goal_object_displacement",
        "home_time",
        "home_success",
        "ft_torque",
        "action_rate",
        "joint_velocity",
        "commanded_effort",
        "torque_saturation",
        "failure_termination",
    ]
    assert not {"reaching", "contact", "push", "home_return"} & set(rewards)


def test_detailed_rewards_use_current_three_phase_contract() -> None:
    source = (PACKAGE / "mdp" / "rewards_detailed.py").read_text(encoding="utf-8")
    assert "PHASE_REACH" in source
    assert "PHASE_SWEEP" in source
    assert "PHASE_HOME" in source
    assert "task_phase == 1" not in source
    assert "task_phase == PHASE_HOME" in source
    assert "TARGET_SIZE_BUFFER" in source
    assert "filtered_contact_mask(env, sensor_names" in source


def test_current_termination_and_sensor_contracts_are_inherited() -> None:
    config_source = (PACKAGE / "env_cfg_detailed.py").read_text(encoding="utf-8")
    assert "TerminationsCfg" not in config_source
    assert "ROBOT_CONTACT_SENSORS" in config_source
    assert '"self_collision"' in config_source
    assert '"shelf_collision"' in config_source
    assert '"contact_lost"' in config_source
    assert '"object_disturbed_home"' in config_source
    assert '"{ENV_REGEX_NS}/Robot/.*"' not in config_source


def test_detailed_document_explains_adaptation_and_training() -> None:
    document = (
        ROOT / "src" / "sweep_rl" / "docs" / "independent_osc_sweep_detailed.md"
    ).read_text(encoding="utf-8")
    assert "참고 환경에서 차용한 부분" in document
    assert "그대로 가져오지 않은 부분" in document
    assert "## REACH reward" in document
    assert "## SWEEP reward와 penalty" in document
    assert "## HOME reward와 penalty" in document
    assert "## Termination" in document
    assert "train_independent_sweep_detailed.py" in document
