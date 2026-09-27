"""Static inheritance and wiring contracts for the approach task variant."""

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


def test_new_environment_and_configs_inherit_parent_task() -> None:
    env_tree = _tree("env_approach.py")
    assert _base_names(_class(env_tree, "BlindSweepApproachEnv")) == {"BlindSweepEnv"}

    cfg_tree = _tree("env_approach_cfg.py")
    assert _base_names(_class(cfg_tree, "BlindSweepApproachTaskCfg")) == {
        "BlindSweepTaskCfg"
    }
    assert _base_names(_class(cfg_tree, "BlindSweepApproachRewardsCfg")) == {"RewardsCfg"}
    assert _base_names(_class(cfg_tree, "BlindSweepApproachEnvCfg")) == {"BlindSweepEnvCfg"}


def test_orientation_aware_reset_is_inherited_and_wired_only_to_variant() -> None:
    event_tree = _tree("mdp/approach_events.py")
    reset = _class(event_tree, "OrientationAwareStableOffsetPoseReset")
    assert _base_names(reset) == {"StableOffsetPoseReset"}
    reset_source = ast.unparse(reset)
    assert "super()._sample_stable_pose" in reset_source
    assert "env.safe_approach_height_offset_m" in reset_source

    cfg_source = ast.unparse(_class(_tree("env_approach_cfg.py"), "BlindSweepApproachEnvCfg"))
    assert "self.events.manipulator_reset.func = OrientationAwareStableOffsetPoseReset" in cfg_source
    base_cfg_source = (PACKAGE_ROOT / "env_cfg.py").read_text(encoding="utf-8")
    assert "OrientationAwareStableOffsetPoseReset" not in base_cfg_source


def test_variant_height_and_reward_defaults_are_explicit() -> None:
    cfg = _class(_tree("env_approach_cfg.py"), "BlindSweepApproachTaskCfg")
    values = {
        node.target.id: ast.literal_eval(node.value)
        for node in cfg.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    assert values["stable_reset_position_offset_task"] == (0.140, 0.0, 0.100)
    assert values["stable_reset_hand_x_up_height_m"] == 0.065
    assert values["stable_reset_hand_x_down_height_m"] == 0.100
    assert values["approach_planar_capture_radius_m"] == 0.015
    assert values["approach_planar_sigma_m"] == 0.080
    assert values["approach_height_sigma_m"] == 0.030

    rewards = ast.unparse(_class(_tree("env_approach_cfg.py"), "BlindSweepApproachRewardsCfg"))
    assert "selected_surface_object_approach" in rewards
    assert "weight=0.1" in rewards


def test_variant_has_separate_registration_runner_and_script_selection() -> None:
    init_source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert 'APPROACH_TASK_ID = "Isaac-Blind-Sweep-Inspire-Approach-v0"' in init_source
    assert 'entry_point="hand_manipulation_rl.env_approach:BlindSweepApproachEnv"' in init_source
    assert "hand_manipulation_rl.env_approach_cfg:BlindSweepApproachEnvCfg" in init_source
    assert "BlindSweepApproachPPORunnerCfg" in init_source

    runner_source = (PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg_approach.py").read_text(
        encoding="utf-8"
    )
    assert "class BlindSweepApproachPPORunnerCfg(BlindSweepReferencePPORunnerCfg)" in runner_source
    assert 'experiment_name = "UR5e_shelf_sweep_approach"' in runner_source

    for script_name in ("train.py", "play.py", "smoke_env.py"):
        script_source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "--task" in script_source
        assert "Isaac-Blind-Sweep-Inspire-Approach-v0" in script_source
