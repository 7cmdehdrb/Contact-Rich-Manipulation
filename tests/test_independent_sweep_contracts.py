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
