"""Guard the GPU filter requests that produced the captured startup warnings.

This replays path matching from the warning, not the unavailable GPU backend.
Independent contact behavior is covered by the production-reader tests.
"""

from fnmatch import fnmatchcase

from test_omni_configuration import omni_config
from test_tactile_configuration import tactile_config


def test_captured_unsupported_colliders_are_not_requested_as_gpu_filters(tactile_config):
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    # These Shelf paths are taken from the user's GPU warning. Static ground
    # filtering has the same limitation on the relevant tensor backend.
    unsupported = ["/World/envs/env_2009/Shelf", "/World/envs/env_2047/Shelf", "/World/ground/Environment/Plane"]
    attempts = []
    for parent in tactile_config.geometry.TACTILE_PARENT_BODY_NAMES:
        sensor = getattr(cfg.scene, tactile_config.geometry.parent_sensor_name(parent))
        for expr in sensor.filter_prim_paths_expr:
            glob = expr.replace("{ENV_REGEX_NS}", "/World/envs/env_.*").replace(".*", "*")
            attempts.extend((parent, path) for path in unsupported if fnmatchcase(path, glob))
    assert attempts == [], f"Unsupported GPU filter requests from the captured warning: {attempts}"
    # Keep supported object contacts for all 17 virtual pad regions.
    for parent in tactile_config.geometry.TACTILE_PARENT_BODY_NAMES:
        sensor = getattr(cfg.scene, tactile_config.geometry.parent_sensor_name(parent))
        assert sensor.filter_prim_paths_expr == ["{ENV_REGEX_NS}/mass_cylinder"]


def test_tactile_debug_markers_do_not_create_the_reported_point_instancer(tactile_config):
    cfg = tactile_config.cfg.InspireSweepTactileEnvCfg()
    baseline = tactile_config.baseline.InspireSweepEnvCfg()
    assert cfg.scene.ee_frame.debug_vis is False
    assert baseline.scene.ee_frame.debug_vis is True
    # Turning off visualization must preserve the transform's physics targets.
    assert cfg.scene.ee_frame.target_frames == baseline.scene.ee_frame.target_frames
