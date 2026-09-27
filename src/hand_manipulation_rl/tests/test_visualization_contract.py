"""Static contract checks for task visualization without launching Isaac Sim."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = (PACKAGE_ROOT / "hand_manipulation_rl" / "env.py").read_text()
CFG_SOURCE = (PACKAGE_ROOT / "hand_manipulation_rl" / "env_cfg.py").read_text()
PLAY_SOURCE = (PACKAGE_ROOT / "scripts" / "play.py").read_text()


def test_training_default_keeps_cpu_side_markers_disabled():
    assert "debug_vis: bool = False" in CFG_SOURCE


def test_markers_follow_live_target_goal_and_virtual_eef():
    assert 'target = self.scene["target_object"]' in ENV_SOURCE
    assert "target.data.root_pos_w, self.goal_pos_w" in ENV_SOURCE
    assert "c_pos_w, c_quat_w = self.control_point_pose_w()" in ENV_SOURCE
    assert 'prim_path="/Visuals/BlindSweep/TaskPoints"' in ENV_SOURCE
    assert 'prim_path="/Visuals/BlindSweep/VirtualEEF"' in ENV_SOURCE


def test_play_enables_markers_by_default_with_opt_out():
    assert '"--disable-markers"' in PLAY_SOURCE
    assert "env_cfg.debug_vis = not args.disable_markers" in PLAY_SOURCE
