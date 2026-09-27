"""Static contract for the active fast, collision-certified reset."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_rl"


def _class_source(path: Path, class_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.unparse(node)


def test_active_event_uses_new_reset_and_keeps_legacy_unregistered():
    cfg_source = _class_source(PACKAGE_ROOT / "env_cfg.py", "EventsCfg")
    assert "func=StableOffsetPoseReset" in cfg_source
    assert "ConditionalPoseIKReset" not in cfg_source


def test_stable_reset_has_no_physics_forward_inside_ik():
    reset_source = _class_source(PACKAGE_ROOT / "mdp" / "events.py", "StableOffsetPoseReset")
    assert "sim.forward" not in reset_source
    assert "reset_clearance_mask" in reset_source
    assert "reset_collision_free_mask" in reset_source
    assert "position_offset_task" in reset_source
    assert "orientation_offset_rpy" in reset_source


def test_legacy_reset_implementation_is_retained():
    event_source = (PACKAGE_ROOT / "mdp" / "events.py").read_text(encoding="utf-8")
    assert "class ConditionalPoseIKReset(ManagerTermBase):" in event_source
