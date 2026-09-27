"""Package-local UR5e--Axia80--Inspire Hand articulation.

The checked-in USD contains UR5e and Inspire geometry.  At spawn time the
original direct wrist/hand fixed joint is replaced with an explicit cylindrical
F/T body and two fixed joints.  This keeps the measurement joint in the
articulation (fixed joints must not be merged) without depending on another
project package, a ROS installation, Nucleus, or a machine-specific path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

HAND_JOINT_NAMES = (
    "inspire_left_thumb_1_joint",
    "inspire_left_thumb_2_joint",
    "inspire_left_thumb_3_joint",
    "inspire_left_thumb_4_joint",
    "inspire_left_index_1_joint",
    "inspire_left_index_2_joint",
    "inspire_left_middle_1_joint",
    "inspire_left_middle_2_joint",
    "inspire_left_ring_1_joint",
    "inspire_left_ring_2_joint",
    "inspire_left_little_1_joint",
    "inspire_left_little_2_joint",
)

HAND_OPEN_TARGETS = {
    "inspire_left_thumb_1_joint": 0.2,
    "inspire_left_thumb_2_joint": 0.0,
    "inspire_left_thumb_3_joint": 0.0,
    "inspire_left_thumb_4_joint": 0.0,
    "inspire_left_index_1_joint": 0.0,
    "inspire_left_index_2_joint": 0.0,
    "inspire_left_middle_1_joint": 0.0,
    "inspire_left_middle_2_joint": 0.0,
    "inspire_left_ring_1_joint": 0.0,
    "inspire_left_ring_2_joint": 0.0,
    "inspire_left_little_1_joint": 0.0,
    "inspire_left_little_2_joint": 0.0,
}

HAND_CLOSED_TARGETS = {
    "inspire_left_thumb_1_joint": 1.1641,
    "inspire_left_thumb_2_joint": 0.5864,
    "inspire_left_thumb_3_joint": 0.47052736,
    "inspire_left_thumb_4_joint": 0.446389306432,
    "inspire_left_index_1_joint": 1.4381,
    "inspire_left_index_2_joint": 1.55933183,
    "inspire_left_middle_1_joint": 1.4381,
    "inspire_left_middle_2_joint": 1.55933183,
    "inspire_left_ring_1_joint": 1.4381,
    "inspire_left_ring_2_joint": 1.55933183,
    "inspire_left_little_1_joint": 1.4381,
    "inspire_left_little_2_joint": 1.55933183,
}

PALM_SENSOR_BODY_NAMES = (
    "inspire_palm_force_sensor",
    *(f"inspire_thumb_force_sensor_{index}" for index in range(1, 5)),
    *(
        f"inspire_{finger}_force_sensor_{index}"
        for finger in ("index", "middle", "ring", "little")
        for index in range(1, 4)
    ),
)

ARM_BODY_NAMES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)

HAND_LINK_BODY_NAMES = (
    "inspire_base_link",
    *(f"inspire_left_thumb_{index}" for index in range(1, 5)),
    *(f"inspire_left_{finger}_{index}" for finger in ("index", "middle", "ring", "little") for index in range(1, 3)),
)

HAND_CONTACT_BODY_NAMES = (*HAND_LINK_BODY_NAMES, *PALM_SENSOR_BODY_NAMES)
ROBOT_CONTACT_BODY_NAMES = (*ARM_BODY_NAMES, "axia80_link", *HAND_CONTACT_BODY_NAMES)

HAND_BASE_BODY_NAME = "inspire_base_link"
WRIST_BODY_NAME = "wrist_3_link"
FT_SENSOR_BODY_NAME = "axia80_link"
FT_MEASUREMENT_CHILD_BODY_NAME = HAND_BASE_BODY_NAME
ORIGINAL_MOUNT_JOINT_NAME = "inspire_mount"

ASSET_ROOT = Path(__file__).resolve().parent / "data" / "ur5e_inspire_usd"
ROBOT_USD_PATH = ASSET_ROOT / "ur5e_inspire.usd"


def _world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def _offset_matrix(translation: tuple[float, float, float]) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def _matrix_to_pose(matrix: Gf.Matrix4d) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat().GetNormalized()
    imaginary = quaternion.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (float(quaternion.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )


def _set_world_transform(prim: Usd.Prim, world_matrix: Gf.Matrix4d) -> None:
    parent_world = _world_transform(prim.GetParent())
    local_matrix = world_matrix * parent_world.GetInverse()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_matrix)
    xformable.SetResetXformStack(False)


def _create_fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    parent_body: Usd.Prim,
    child_body: Usd.Prim,
    joint_frame_world: Gf.Matrix4d,
) -> None:
    parent_local = joint_frame_world * _world_transform(parent_body).GetInverse()
    child_local = joint_frame_world * _world_transform(child_body).GetInverse()
    parent_pos, parent_rot = _matrix_to_pose(parent_local)
    child_pos, child_rot = _matrix_to_pose(child_local)
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody0Rel().SetTargets([parent_body.GetPath()])
    joint.CreateBody1Rel().SetTargets([child_body.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*parent_pos))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(parent_rot[0], Gf.Vec3f(*parent_rot[1:])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pos))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(child_rot[0], Gf.Vec3f(*child_rot[1:])))
    joint.CreateCollisionEnabledAttr().Set(False)


def _move_hand_rigid_bodies(stage: Usd.Stage, robot_path: str, desired_base_world: Gf.Matrix4d) -> Usd.Prim:
    hand_base = stage.GetPrimAtPath(f"{robot_path}/{HAND_BASE_BODY_NAME}")
    if not hand_base.IsValid() or not hand_base.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Missing hand base rigid body: {hand_base.GetPath()}")
    old_base_world = _world_transform(hand_base)
    bodies = [
        prim
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
        if prim.GetName().startswith("inspire_") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    old_world = {prim.GetPath(): _world_transform(prim) for prim in bodies}
    for prim in bodies:
        body_relative_to_base = old_world[prim.GetPath()] * old_base_world.GetInverse()
        _set_world_transform(prim, body_relative_to_base * desired_base_world)
    return hand_base


@clone
def spawn_ur5e_axia80_inspire(
    prim_path: str,
    cfg: "Ur5eAxia80InspireSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn one assembled source articulation before Isaac Lab clones it."""

    del kwargs
    if not Path(cfg.usd_path).is_file():
        raise FileNotFoundError(f"Package-local robot USD is missing: {cfg.usd_path}")

    usd_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.usd_path,
        rigid_props=cfg.rigid_props,
        articulation_props=cfg.articulation_props,
        activate_contact_sensors=True,
    )
    usd_cfg.func(prim_path, usd_cfg, translation=translation, orientation=orientation)

    stage = get_current_stage()
    robot_root = stage.GetPrimAtPath(prim_path)
    wrist_body = stage.GetPrimAtPath(f"{prim_path}/{WRIST_BODY_NAME}")
    hand_base = stage.GetPrimAtPath(f"{prim_path}/{HAND_BASE_BODY_NAME}")
    old_mount = stage.GetPrimAtPath(f"{prim_path}/joints/{ORIGINAL_MOUNT_JOINT_NAME}")
    if not robot_root.IsValid() or not wrist_body.IsValid() or not hand_base.IsValid() or not old_mount.IsValid():
        raise RuntimeError("The package-local USD does not match the expected UR5e/Inspire topology.")

    mount_world = _world_transform(hand_base)
    desired_hand_base_world = _offset_matrix((0.0, 0.0, cfg.sensor_height)) * mount_world
    hand_base = _move_hand_rigid_bodies(stage, prim_path, desired_hand_base_world)

    old_joint = UsdPhysics.Joint(old_mount)
    old_joint.GetJointEnabledAttr().Set(False)
    old_mount.SetActive(False)

    sensor_world = _offset_matrix((0.0, 0.0, 0.5 * cfg.sensor_height)) * mount_world
    sensor_pos, sensor_rot = _matrix_to_pose(sensor_world * _world_transform(robot_root).GetInverse())
    sensor_cfg = sim_utils.CylinderCfg(
        radius=cfg.sensor_radius,
        height=cfg.sensor_height,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=cfg.sensor_mass),
        # The fixed joints disable collision only for their directly connected
        # pairs.  Keeping the housing collider active lets reset/safety sensors
        # detect an Axia80--board or Axia80--object collision.
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.001,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.38, 0.08)),
        activate_contact_sensors=True,
    )
    sensor_cfg.func(
        f"{prim_path}/{FT_SENSOR_BODY_NAME}",
        sensor_cfg,
        translation=sensor_pos,
        orientation=sensor_rot,
    )
    sensor_body = stage.GetPrimAtPath(f"{prim_path}/{FT_SENSOR_BODY_NAME}")
    if not sensor_body.IsValid() or not sensor_body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Failed to create the Axia80 measurement body.")

    _create_fixed_joint(
        stage,
        f"{prim_path}/joints/ur5e_to_axia80_joint",
        wrist_body,
        sensor_body,
        mount_world,
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/joints/axia80_measurement_joint",
        sensor_body,
        hand_base,
        desired_hand_base_world,
    )
    return robot_root


@configclass
class Ur5eAxia80InspireSpawnerCfg(RigidObjectSpawnerCfg):
    """Spawner configuration for the package-local assembled robot."""

    func: Callable = spawn_ur5e_axia80_inspire
    activate_contact_sensors: bool = True
    usd_path: str = str(ROBOT_USD_PATH)
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=True,
        max_depenetration_velocity=5.0,
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = sim_utils.ArticulationRootPropertiesCfg(
        # The source Inspire collision meshes overlap at several adjacent
        # knuckles in valid open postures.  PhysX cannot express the required
        # adjacent-pair exclusions through ArticulationCfg, so follow the
        # validated source-hand setup and keep articulation self-collision off.
        # Reset IK still enforces joint/wrist/singularity bounds and applies a
        # conservative robot--board/object clearance proxy.  Exact hand
        # self-collision remains an asset limitation documented in README.
        enabled_self_collisions=False,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=4,
    )
    sensor_radius: float = 0.041
    sensor_height: float = 0.0254
    sensor_mass: float = 0.001


def make_robot_cfg() -> ArticulationCfg:
    """Return the fixed-base effort-arm / position-hand articulation config."""

    initial_arm = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -2.2,
        "elbow_joint": 2.2,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 1.57,
        "wrist_3_joint": 0.785,
    }
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=Ur5eAxia80InspireSpawnerCfg(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.79505),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos=initial_arm | HAND_OPEN_TARGETS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                effort_limit_sim={
                    "shoulder_pan_joint": 150.0,
                    "shoulder_lift_joint": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_1_joint": 28.0,
                    "wrist_2_joint": 28.0,
                    "wrist_3_joint": 28.0,
                },
                velocity_limit_sim={
                    "shoulder_.*": 3.14,
                    "elbow_joint": 3.14,
                    "wrist_.*": 6.28,
                },
                stiffness=0.0,
                damping=0.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(HAND_JOINT_NAMES),
                effort_limit_sim=0.20,
                velocity_limit_sim=1.0,
                stiffness=1.2,
                damping=0.06,
                armature=0.0001,
            ),
        },
    )


def validate_asset_contract() -> None:
    if not ROBOT_USD_PATH.is_file():
        raise FileNotFoundError(ROBOT_USD_PATH)
    if len(ARM_JOINT_NAMES) != 6 or len(HAND_JOINT_NAMES) != 12 or len(PALM_SENSOR_BODY_NAMES) != 17:
        raise ValueError("Unexpected UR5e/Inspire joint or tactile-channel contract.")
    if set(HAND_OPEN_TARGETS) != set(HAND_CLOSED_TARGETS):
        raise ValueError("Open/closed hand maps must cover identical joints.")


validate_asset_contract()


__all__ = [
    "ARM_JOINT_NAMES",
    "ARM_BODY_NAMES",
    "FT_MEASUREMENT_CHILD_BODY_NAME",
    "FT_SENSOR_BODY_NAME",
    "HAND_BASE_BODY_NAME",
    "HAND_CLOSED_TARGETS",
    "HAND_CONTACT_BODY_NAMES",
    "HAND_JOINT_NAMES",
    "HAND_LINK_BODY_NAMES",
    "HAND_OPEN_TARGETS",
    "PALM_SENSOR_BODY_NAMES",
    "ROBOT_CONTACT_BODY_NAMES",
    "ROBOT_USD_PATH",
    "make_robot_cfg",
]
