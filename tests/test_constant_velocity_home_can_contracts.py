"""Static contracts for the Can-based constant-velocity HomeReturn task."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "sweep_rl" / "sweep_rl" / "osc_sweep"
CAN_CONFIG = PACKAGE / "env_cfg_constant_velocity_upright_random_size_home_can.py"
HOME_CONFIG = PACKAGE / "env_cfg_constant_velocity_upright_random_size_home.py"
CAN_PLAYER = ROOT / "src" / "sweep_rl" / "scripts" / "play_constant_velocity_home_can.py"


def test_home_return_uses_can_usd_with_bottom_origin() -> None:
    source = CAN_CONFIG.read_text(encoding="utf-8")
    original_source = HOME_CONFIG.read_text(encoding="utf-8")
    assert (
        'CAN_USD_PATH = "omniverse://192.168.0.13/'
        'Library/Shelf/Objects/Can_6/Can_6.usd"'
    ) in source
    assert "target_object = RigidObjectCfg(" in source
    assert "usd_path=CAN_USD_PATH" in source
    assert "OPEN_TABLE_TOP_HEIGHT = 0.775" in source
    assert "pos=(0.50, 0.0, OPEN_TABLE_TOP_HEIGHT)" in source
    assert "CAN_USD_PATH" not in original_source
    assert 'filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"]' not in original_source
    assert "for body_path in ROBOT_CONTACT_BODY_PATHS" in original_source


def test_can_mass_is_defined_on_rigid_root_during_prestartup() -> None:
    config_source = CAN_CONFIG.read_text(encoding="utf-8")
    event_source = (PACKAGE / "mdp" / "events.py").read_text(encoding="utf-8")

    assert "mass_props=sim_utils.MassPropertiesCfg" not in config_source
    assert "class CanSweepHomeEventsCfg(EventCfg):" in config_source
    assert "set_target_mass = EventTerm(" in config_source
    assert "func=mdp.define_rigid_object_mass" in config_source
    assert 'mode="prestartup"' in config_source
    assert '"mass": DEFAULT_CAN_MASS' in config_source
    assert "sim_utils.define_mass_properties(prim_path, mass_cfg)" in event_source
    assert "Expected {env.scene.num_envs} rigid-object roots" in event_source


def test_can_observation_uses_center_height_without_changing_shape() -> None:
    config_source = CAN_CONFIG.read_text(encoding="utf-8")
    observation_source = (PACKAGE / "mdp" / "observations.py").read_text(
        encoding="utf-8"
    )
    assert "CAN_OBSERVATION_Z_OFFSET = 0.5 * CAN_HEIGHT" in config_source
    assert "func=mdp.initial_target_pose_b_with_z_offset" in config_source
    assert "func=mdp.current_target_pose_b_with_z_offset" in config_source
    assert observation_source.count("pose[:, 2] += z_offset") == 2
    assert "policy: PolicyCfg = PolicyCfg()" in config_source


def test_dedicated_player_and_usage_are_documented() -> None:
    player = CAN_PLAYER.read_text(encoding="utf-8")
    document = (
        ROOT
        / "src"
        / "sweep_rl"
        / "docs"
        / "constant_velocity_upright_random_size_home_return.md"
    ).read_text(encoding="utf-8")
    assert "--object_mass" in player
    assert "--target_z_offset" in player
    assert "HomeReturn-Can-v0" in player
    assert "play_constant_velocity_home_can.py" in document
    assert "--object_mass 1.25" in document
    assert "env.scene.target_object.spawn.mass_props.mass=..." in document
