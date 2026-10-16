"""Production tactile reader on a test fixture with explicit pad placement."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES, PALM_SENSOR_BODY_NAMES
from inspire_sweep_baseline.assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name

from .hand import make_hand_cfg


def make_scene(num_envs, *, spacing, probe_radius, probe_length, pad_layout="original", pad_clearance=0.001):
    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=spacing, replicate_physics=True)
    cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    cfg.robot = make_hand_cfg(pad_layout=pad_layout, pad_clearance=pad_clearance)
    cfg.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        spawn=sim_utils.CapsuleCfg(
            radius=probe_radius, height=probe_length, axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=0.5, sleep_threshold=0.0,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.1)),
            activate_contact_sensors=True,
        ),
    )
    cfg.hand_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")",
        update_period=0.0, history_length=0, force_threshold=0.05,
    )
    for parent in TACTILE_PARENT_BODY_NAMES:
        setattr(cfg, parent_sensor_name(parent), ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + parent,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Probe"],
            update_period=0.0, history_length=0, max_contact_data_count_per_prim=16,
        ))
    # Independent physical-contact reference, irrespective of tactile bits.
    cfg.probe_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        track_contact_points=True, max_contact_data_count_per_prim=64,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + name for name in HAND_CONTACT_BODY_NAMES],
        update_period=0.0, history_length=0,
    )
    return cfg
