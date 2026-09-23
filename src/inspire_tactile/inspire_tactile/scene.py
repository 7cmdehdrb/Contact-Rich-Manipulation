"""Isaac Lab scene. Import only after AppLauncher starts Isaac Sim."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from .placement import (
    DEFAULT_OBJECT_LOWERING,
    DEFAULT_OBJECT_OFFSET,
    object_and_support_positions,
)


def make_scene(
    urdf: Path,
    model: dict,
    *,
    num_envs: int = 1,
    object_offset=DEFAULT_OBJECT_OFFSET,
    object_lowering=DEFAULT_OBJECT_LOWERING,
    without_object=False,
    fix_object=False,
    force_conversion=False,
    tactile_mode="projected",
) -> InteractiveSceneCfg:
    cfg = InteractiveSceneCfg(
        num_envs=num_envs, env_spacing=2.0, replicate_physics=True
    )
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            usd_dir=str(urdf.parent / "usd"),
            usd_file_name="ur5e_inspire.usd",
            force_usd_conversion=force_conversion,
            fix_base=True,
            root_link_name="base_link",
            merge_fixed_joints=False,
            # Explicit motor targets reproduce URDF mimic ratios in this rule-based demo.
            convert_mimic_joints_to_normal_joints=False,
            make_instanceable=False,
            activate_contact_sensors=True,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                )
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=0.5
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=model["arm_pose"] | model["open_targets"]
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(model["arm_pose"]),
                stiffness=8000.0,
                damping=400.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=1.0,
                armature=0.01,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["inspire_left_.*_joint"],
                stiffness=1.2,
                damping=0.06,
                effort_limit_sim=0.20,
                velocity_limit_sim=1.0,
                armature=0.0001,
            ),
        },
    )
    cfg.tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/inspire_.*force_sensor.*",
        update_period=0.0,
        history_length=0,
        track_pose=True,
        # This field controls contact-time tracking, NOT thresholding of net_forces_w.
        force_threshold=0.001,
        debug_vis=True,
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2200.0)
    )
    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.18, 0.22)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )
    radius, height = 0.04, 0.09
    position, support_position = object_and_support_positions(
        model["hand_pose"],
        object_offset,
        object_lowering,
        cylinder_height=height,
    )
    cfg.pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.06, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001, rest_offset=0.0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.30, 0.24)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(x) for x in support_position)
        ),
    )
    if not without_object:
        cfg.cylinder = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cylinder",
            spawn=sim_utils.CylinderCfg(
                radius=radius,
                height=height,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=fix_object,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.32),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.001, rest_offset=0.0
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=0.8, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.35, 0.08)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(x) for x in position)
            ),
        )
    if tactile_mode not in ("projected", "raw"):
        raise ValueError(f"Unknown tactile mode: {tactile_mode}")
    if tactile_mode == "projected":
        from .regions import load_regions, observer_key
        # Contact reports are already enabled on ALL robot bodies above. These
        # additional read-only views do not create/resize colliders or joints.
        # This GPU tensor backend does not support static collider filters.
        # Keep pedestal/ground STATIC: do not change physics just for readout.
        # Their parent-link net forces remain unassigned diagnostics.
        filters = [] if without_object else ["{ENV_REGEX_NS}/Cylinder"]
        for parent in load_regions(urdf, model["sensor_names"]):
            setattr(cfg, observer_key(parent), ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + parent,
                update_period=0.0, history_length=0, debug_vis=False,
                filter_prim_paths_expr=filters,
                max_contact_data_count_per_prim=1024,
            ))
    return cfg
