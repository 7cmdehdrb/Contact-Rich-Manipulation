"""Sensor-free UR5e and Robotiq 2F-85 articulation assembly."""

from __future__ import annotations

import os
from collections.abc import Callable

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
GRIPPER_JOINT_PATTERN = ".*(finger|knuckle).*"
ROBOT_BASE_BODY_PATH = "base_link"
GRIPPER_BASE_BODY_PATH = "Robotiq2F85/Robotiq_2F_85/base_link"
LEFT_FINGER_BODY_PATH = "Robotiq2F85/Robotiq_2F_85/left_outer_finger"
RIGHT_FINGER_BODY_PATH = "Robotiq2F85/Robotiq_2F_85/right_outer_finger"
WRIST_BODY_PATH = "wrist_3_link"

DEFAULT_UR5E_USD_PATH = (
    "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.0/"
    "Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
)
DEFAULT_ROBOTIQ_USD_PATH = (
    "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.1/"
    "Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
)

_UR_TOOL_FRAME_CANDIDATES = ("tool0", "tool_frame", "flange", "wrist_3_link")
_GRIPPER_BASE_BODY_CANDIDATES = (
    "robotiq_arg2f_base_link",
    "robotiq_2f_85_base_link",
    "robotiq_base_link",
    "base_link",
)
_PREEXISTING_MOUNT_JOINT_NAMES = ("robot_gripper_joint",)


def _world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def _find_named_prim(
    stage: Usd.Stage,
    subtree_path: str,
    candidates: tuple[str, ...],
    *,
    rigid_body_only: bool = False,
) -> Usd.Prim:
    root = stage.GetPrimAtPath(subtree_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid asset subtree: {subtree_path}")
    for candidate in candidates:
        for prim in Usd.PrimRange(root):
            if prim.GetName() != candidate:
                continue
            if rigid_body_only and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            return prim
    raise RuntimeError(
        f"Could not find any of {candidates} below '{subtree_path}'."
    )


def _nearest_rigid_body_ancestor(prim: Usd.Prim) -> Usd.Prim:
    current = prim
    while current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    raise RuntimeError(f"No rigid-body ancestor found for '{prim.GetPath()}'.")


def _make_offset_matrix(
    translation: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    rotation = (
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rotation_deg[0])
        * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), rotation_deg[1])
        * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_deg[2])
    )
    matrix.SetRotate(rotation)
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def _matrix_to_pose(
    matrix: Gf.Matrix4d,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat().GetNormalized()
    imaginary = quaternion.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _set_world_transform(root_prim: Usd.Prim, world_matrix: Gf.Matrix4d) -> None:
    local_matrix = world_matrix * _world_transform(root_prim.GetParent()).GetInverse()
    xformable = UsdGeom.Xformable(root_prim)
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
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(parent_rot[0], Gf.Vec3f(*parent_rot[1:]))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pos))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(child_rot[0], Gf.Vec3f(*child_rot[1:]))
    )
    joint.CreateCollisionEnabledAttr().Set(False)


def _deactivate_old_mount_joints(stage: Usd.Stage, robot_path: str) -> None:
    root = stage.GetPrimAtPath(robot_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() not in _PREEXISTING_MOUNT_JOINT_NAMES:
            continue
        joint = UsdPhysics.Joint(prim)
        if joint:
            joint.GetJointEnabledAttr().Set(False)
            prim.SetActive(False)


def _remove_nested_articulation_roots(stage: Usd.Stage, subtree_path: str) -> None:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(subtree_path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _deinstance_geometry(stage: Usd.Stage, subtree_path: str) -> None:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(subtree_path)):
        if prim.IsInstance():
            prim.SetInstanceable(False)


def _sanitize_collision_apis(stage: Usd.Stage, subtree_path: str) -> None:
    geometry_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
    )
    for prim in Usd.PrimRange(stage.GetPrimAtPath(subtree_path)):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not any(
            prim.IsA(geometry_type) for geometry_type in geometry_types
        ):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)


@clone
def spawn_sensorless_ur5e_robotiq(
    prim_path: str,
    cfg: "SensorlessUr5eRobotiqSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Compose the arm and gripper into one articulation before cloning."""
    del kwargs
    stage = get_current_stage()
    ur_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.ur5e_usd_path,
        rigid_props=cfg.rigid_props,
        articulation_props=cfg.articulation_props,
        activate_contact_sensors=False,
    )
    ur_cfg.func(
        prim_path, ur_cfg, translation=translation, orientation=orientation
    )

    gripper_path = f"{prim_path}/Robotiq2F85"
    gripper_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.robotiq_usd_path,
        activate_contact_sensors=False,
    )
    gripper_cfg.func(gripper_path, gripper_cfg)

    robot_root = stage.GetPrimAtPath(prim_path)
    gripper_root = stage.GetPrimAtPath(gripper_path)
    if not robot_root.IsValid() or not gripper_root.IsValid():
        raise RuntimeError(
            "UR5e or Robotiq USD did not compose. Check SWEEP_UR5E_USD_PATH "
            "and SWEEP_ROBOTIQ_USD_PATH."
        )

    _deinstance_geometry(stage, prim_path)
    tool_frame = _find_named_prim(stage, prim_path, cfg.tool_frame_candidates)
    tool_body = _nearest_rigid_body_ancestor(tool_frame)
    gripper_base = _find_named_prim(
        stage,
        gripper_path,
        cfg.gripper_base_candidates,
        rigid_body_only=True,
    )

    _deactivate_old_mount_joints(stage, prim_path)
    root_world = _world_transform(gripper_root)
    base_world = _world_transform(gripper_base)
    base_relative_to_root = base_world * root_world.GetInverse()
    tool_world = _world_transform(tool_frame)
    desired_base_world = (
        _make_offset_matrix(cfg.mount_translation, cfg.mount_rotation_deg)
        * tool_world
    )
    desired_root_world = base_relative_to_root.GetInverse() * desired_base_world
    _set_world_transform(gripper_root, desired_root_world)

    _remove_nested_articulation_roots(stage, gripper_path)
    _sanitize_collision_apis(stage, prim_path)
    _sanitize_collision_apis(stage, gripper_path)
    _create_fixed_joint(
        stage,
        f"{prim_path}/UR5e_Robotiq_mount_joint",
        tool_body,
        gripper_base,
        desired_base_world,
    )
    return robot_root


@configclass
class SensorlessUr5eRobotiqSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for a direct UR5e-to-Robotiq assembly."""

    func: Callable = spawn_sensorless_ur5e_robotiq
    ur5e_usd_path: str = DEFAULT_UR5E_USD_PATH
    robotiq_usd_path: str = DEFAULT_ROBOTIQ_USD_PATH
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=0.5,
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = (
        sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        )
    )
    tool_frame_candidates: tuple[str, ...] = _UR_TOOL_FRAME_CANDIDATES
    gripper_base_candidates: tuple[str, ...] = _GRIPPER_BASE_BODY_CANDIDATES
    mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rotation_deg: tuple[float, float, float] = (0.0, 90.0, 0.0)


def make_sensorless_robot_cfg() -> ArticulationCfg:
    """Build the position-controlled robot configuration."""
    spawn_cfg = SensorlessUr5eRobotiqSpawnerCfg(
        ur5e_usd_path=os.environ.get("SWEEP_UR5E_USD_PATH", DEFAULT_UR5E_USD_PATH),
        robotiq_usd_path=os.environ.get(
            "SWEEP_ROBOTIQ_USD_PATH", DEFAULT_ROBOTIQ_USD_PATH
        ),
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.79505),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.2,
                "elbow_joint": 2.2,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.57,
                "wrist_3_joint": 0.785,
                GRIPPER_JOINT_PATTERN: 0.0,
            },
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
                    "shoulder_pan_joint": 3.14,
                    "shoulder_lift_joint": 3.14,
                    "elbow_joint": 3.14,
                    "wrist_1_joint": 6.28,
                    "wrist_2_joint": 6.28,
                    "wrist_3_joint": 6.28,
                },
                stiffness={
                    "shoulder_pan_joint": 3328.7,
                    "shoulder_lift_joint": 4989.4,
                    "elbow_joint": 2394.2,
                    "wrist_1_joint": 3500.2,
                    "wrist_2_joint": 3174.8,
                    "wrist_3_joint": 3032.4,
                },
                damping={
                    "shoulder_pan_joint": 368.8,
                    "shoulder_lift_joint": 561.2,
                    "elbow_joint": 274.1,
                    "wrist_1_joint": 396.2,
                    "wrist_2_joint": 363.1,
                    "wrist_3_joint": 345.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[GRIPPER_JOINT_PATTERN],
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=2000.0,
                damping=1000.0,
            ),
        },
    )
