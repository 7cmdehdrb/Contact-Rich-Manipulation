"""Isaac Lab scene configuration.  Import only after AppLauncher starts Kit."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg

from .model import ARM_JOINT_NAMES, PAYLOAD_SIZE_M


def make_scene(
    urdf_path: Path,
    model: dict[str, object],
    *,
    num_envs: int,
    env_spacing: float,
    force_conversion: bool,
) -> InteractiveSceneCfg:
    cfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
    )
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf_path),
            usd_dir=str(urdf_path.parent / "usd"),
            usd_file_name="ur5e_axia80_tool.usd",
            force_usd_conversion=force_conversion,
            fix_base=True,
            root_link_name="base_link",
            merge_fixed_joints=False,
            make_instanceable=False,
            activate_contact_sensors=False,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.25,
                retain_accelerations=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(model["arm_pose"]),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                stiffness=8000.0,
                damping=400.0,
                effort_limit_sim={
                    "shoulder_pan_joint": 150.0,
                    "shoulder_lift_joint": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_1_joint": 28.0,
                    "wrist_2_joint": 28.0,
                    "wrist_3_joint": 28.0,
                },
                velocity_limit_sim=1.0,
                armature=0.01,
            )
        },
    )
    cfg.payload = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Payload",
        spawn=sim_utils.CuboidCfg(
            size=(PAYLOAD_SIZE_M, PAYLOAD_SIZE_M, PAYLOAD_SIZE_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.20,
                angular_damping=0.50,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.25,
                retain_accelerations=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.010),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.78, 0.12),
                metallic=0.05,
                roughness=0.55,
            ),
        ),
        # Parked above each environment until the runner disables payload
        # gravity and explicitly places every cube on its corresponding tool.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            )
        ),
    )
    cfg.light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    return cfg
