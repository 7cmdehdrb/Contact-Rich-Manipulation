"""Static wiring contracts for the corrected Approach-v1 environment."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "hand_manipulation_rl"


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _base_names(node: ast.ClassDef) -> set[str]:
    return {
        base.id if isinstance(base, ast.Name) else ast.unparse(base)
        for base in node.bases
    }


def test_v1_environment_and_configs_inherit_approach_v0() -> None:
    assert _base_names(_class(_tree("env_v1.py"), "BlindSweepApproachV1Env")) == {
        "BlindSweepApproachEnv"
    }
    cfg_tree = _tree("env_v1_cfg.py")
    assert _base_names(_class(cfg_tree, "BlindSweepApproachV1TaskCfg")) == {
        "BlindSweepApproachTaskCfg"
    }
    assert _base_names(_class(cfg_tree, "BlindSweepApproachV1RewardsCfg")) == {
        "BlindSweepApproachRewardsCfg"
    }
    assert _base_names(_class(cfg_tree, "BlindSweepApproachV1EnvCfg")) == {
        "BlindSweepApproachEnvCfg"
    }


def test_v1_fixes_controller_reward_and_runner_failure_modes() -> None:
    cfg_source = ast.unparse(_class(_tree("env_v1_cfg.py"), "BlindSweepApproachV1EnvCfg"))
    assert "gravity_compensation = True" in cfg_source
    assert "TaskFrameOscAction" in cfg_source
    assert "contact_push_force_n" in cfg_source

    task_source = ast.unparse(_class(_tree("env_v1_cfg.py"), "BlindSweepApproachV1TaskCfg"))
    assert "(0.004, 0.012, 0.004)" in task_source
    assert "(0.1, 0.0, 0.1)" in task_source
    assert "approach_contact_push_force_n: float = 10.0" in task_source

    rewards_source = ast.unparse(_class(_tree("env_v1_cfg.py"), "BlindSweepApproachV1RewardsCfg"))
    assert "SelectedSurfaceApproachProgress" in rewards_source
    assert "selected_surface_approach = None" in rewards_source
    assert "tactile_contact = None" in rewards_source
    assert "TargetHandObjectContactAcquisition" in rewards_source
    assert "selected_surface_height_safety" in rewards_source

    runner_source = (PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg_v1.py").read_text(
        encoding="utf-8"
    )
    assert 'algorithm.schedule = "fixed"' in runner_source
    assert "algorithm.learning_rate = 3.0e-4" in runner_source
    assert "actor.distribution_cfg.init_std = 0.5" in runner_source


def test_v1_has_separate_registration_and_script_selection() -> None:
    init_source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert 'APPROACH_V1_TASK_ID = "Isaac-Blind-Sweep-Inspire-Approach-v1"' in init_source
    assert "hand_manipulation_rl.env_v1:BlindSweepApproachV1Env" in init_source
    assert "BlindSweepApproachV1PPORunnerCfg" in init_source

    for script_name in ("train.py", "play.py", "smoke_env.py"):
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "Isaac-Blind-Sweep-Inspire-Approach-v1" in source
        assert "BlindSweepApproachV1EnvCfg" in source
