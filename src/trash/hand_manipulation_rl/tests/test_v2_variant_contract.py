"""Static reward and wiring contracts for the Approach-v2 environment."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "hand_manipulation_rl"


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _base_names(node: ast.ClassDef) -> set[str]:
    return {
        base.id if isinstance(base, ast.Name) else ast.unparse(base)
        for base in node.bases
    }


def test_v2_inherits_the_complete_v1_environment() -> None:
    assert _base_names(_class(_tree("env_v2.py"), "BlindSweepApproachV2Env")) == {
        "BlindSweepApproachV1Env"
    }
    cfg_tree = _tree("env_v2_cfg.py")
    assert _base_names(_class(cfg_tree, "BlindSweepApproachV2RewardsCfg")) == {
        "BlindSweepApproachV1RewardsCfg"
    }
    assert _base_names(_class(cfg_tree, "BlindSweepApproachV2EnvCfg")) == {
        "BlindSweepApproachV1EnvCfg"
    }


def test_v2_only_rebalances_goal_alignment_and_success_coefficients() -> None:
    rewards = ast.unparse(
        _class(_tree("env_v2_cfg.py"), "BlindSweepApproachV2RewardsCfg")
    )
    assert "object_goal_reward, weight=12.0" in rewards
    assert "actual_contact_normal_alignment, weight=4.0" in rewards
    assert "success_terminal_bonus_rate, weight=4.0" in rewards
    assert "selected_surface_approach_progress" not in rewards
    assert "target_hand_contact_acquisition" not in rewards


def test_v2_has_separate_registration_runner_and_script_selection() -> None:
    init_source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert 'APPROACH_V2_TASK_ID = "Isaac-Blind-Sweep-Inspire-Approach-v2"' in init_source
    assert "hand_manipulation_rl.env_v2:BlindSweepApproachV2Env" in init_source
    assert "BlindSweepApproachV2PPORunnerCfg" in init_source

    runner_source = (PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg_v2.py").read_text(
        encoding="utf-8"
    )
    assert "BlindSweepApproachV1PPORunnerCfg" in runner_source
    assert 'experiment_name = "UR5e_shelf_sweep_approach_v2"' in runner_source

    for script_name in ("train.py", "play.py", "smoke_env.py"):
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "Isaac-Blind-Sweep-Inspire-Approach-v2" in source
        assert "BlindSweepApproachV2EnvCfg" in source
