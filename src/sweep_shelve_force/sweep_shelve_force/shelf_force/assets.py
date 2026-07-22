"""Independent UR5e--virtual F/T--Robotiq articulation assembly."""

from __future__ import annotations

from collections.abc import Callable

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from .asset_manifest import ROBOTIQ_USD_PATH, UR5E_USD_PATH


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
FT_SENSOR_BODY_NAME = "VirtualFTSensor"
EEF_CENTER_BODY_NAME = "SweepToolCenter"

_TOOL_FRAME_CANDIDATES = ("tool0", "tool_frame", "flange", "wrist_3_link")
_GRIPPER_BASE_CANDIDATES = (
    "robotiq_arg2f_base_link",
    "robotiq_2f_85_base_link",
    "robotiq_base_link",
    "base_link",
)


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
        raise RuntimeError(f"Invalid USD subtree: {subtree_path}")
    prims = list(Usd.PrimRange(root))
    for name in candidates:
        for prim in prims:
            if prim.GetName() != name:
                continue
            if rigid_body_only and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            return prim
    available = [prim.GetName() for prim in prims]
    raise RuntimeError(
        f"Could not find any of {candidates} below {subtree_path}. "
        f"Available prims: {available}"
    )


def _nearest_rigid_body(prim: Usd.Prim) -> Usd.Prim:
    current = prim
    while current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    raise RuntimeError(f"No rigid-body ancestor for {prim.GetPath()}")


def _offset_matrix(
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


def _set_reference_root_world_transform(root: Usd.Prim, world: Gf.Matrix4d) -> None:
    local = world * _world_transform(root.GetParent()).GetInverse()
    xform = UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local)
    xform.SetResetXformStack(False)


def _create_fixed_joint(
    stage: Usd.Stage,
    path: str,
    parent: Usd.Prim,
    child: Usd.Prim,
    frame_world: Gf.Matrix4d,
) -> None:
    parent_pos, parent_quat = _matrix_to_pose(
        frame_world * _world_transform(parent).GetInverse()
    )
    child_pos, child_quat = _matrix_to_pose(
        frame_world * _world_transform(child).GetInverse()
    )
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(path))
    joint.CreateBody0Rel().SetTargets([parent.GetPath()])
    joint.CreateBody1Rel().SetTargets([child.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*parent_pos))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(parent_quat[0], Gf.Vec3f(*parent_quat[1:]))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pos))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(child_quat[0], Gf.Vec3f(*child_quat[1:]))
    )
    joint.CreateCollisionEnabledAttr().Set(False)


def _spawn_virtual_body(
    stage: Usd.Stage,
    robot_path: str,
    name: str,
    world: Gf.Matrix4d,
    size: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
) -> Usd.Prim:
    local = world * _world_transform(stage.GetPrimAtPath(robot_path)).GetInverse()
    translation, orientation = _matrix_to_pose(local)
    body_path = f"{robot_path}/{name}"
    cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(body_path, cfg, translation=translation, orientation=orientation)
    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Failed to create virtual body: {body_path}")
    return body


def _prepare_referenced_gripper(stage: Usd.Stage, subtree_path: str) -> None:
    root = stage.GetPrimAtPath(subtree_path)
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _deactivate_preexisting_mount_joints(stage: Usd.Stage, robot_path: str) -> None:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.GetName() == "robot_gripper_joint":
            joint = UsdPhysics.Joint(prim)
            if joint:
                joint.GetJointEnabledAttr().Set(False)
            prim.SetActive(False)


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
def spawn_shelf_force_robot(
    prim_path: str,
    cfg: "ShelfForceRobotSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Compose one robot; Isaac Lab clones the completed articulation."""
    del kwargs
    stage = get_current_stage()

    arm_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.ur5e_usd_path,
        rigid_props=cfg.rigid_props,
        articulation_props=cfg.articulation_props,
        activate_contact_sensors=False,
    )
    arm_cfg.func(
        prim_path,
        arm_cfg,
        translation=translation,
        orientation=orientation,
    )

    gripper_path = f"{prim_path}/Robotiq2F85"
    gripper_cfg = sim_utils.UsdFileCfg(usd_path=cfg.robotiq_usd_path)
    gripper_cfg.func(gripper_path, gripper_cfg)

    robot_root = stage.GetPrimAtPath(prim_path)
    gripper_root = stage.GetPrimAtPath(gripper_path)
    if not robot_root.IsValid() or not gripper_root.IsValid():
        raise RuntimeError(
            "UR5e or Robotiq USD did not load. Check the shelf-force USD environment variables."
        )

    _prepare_referenced_gripper(stage, gripper_path)
    tool_frame = _find_named_prim(stage, prim_path, cfg.tool_frame_candidates)
    tool_body = _nearest_rigid_body(tool_frame)
    gripper_base = _find_named_prim(
        stage,
        gripper_path,
        cfg.gripper_base_candidates,
        rigid_body_only=True,
    )

    root_world = _world_transform(gripper_root)
    base_world = _world_transform(gripper_base)
    base_from_root = base_world * root_world.GetInverse()
    sensor_world = _world_transform(tool_frame)
    desired_base_world = (
        _offset_matrix(cfg.mount_translation, cfg.mount_rotation_deg) * sensor_world
    )
    desired_root_world = base_from_root.GetInverse() * desired_base_world
    _set_reference_root_world_transform(gripper_root, desired_root_world)
    _deactivate_preexisting_mount_joints(stage, prim_path)
    _sanitize_collision_apis(stage, prim_path)

    ft_body = _spawn_virtual_body(
        stage,
        prim_path,
        FT_SENSOR_BODY_NAME,
        sensor_world,
        cfg.ft_sensor_size,
        cfg.ft_sensor_mass,
        (0.95, 0.55, 0.10),
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/tool_to_virtual_ft_joint",
        tool_body,
        ft_body,
        sensor_world,
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/virtual_ft_to_gripper_joint",
        ft_body,
        gripper_base,
        sensor_world,
    )

    center_world = _offset_matrix(cfg.eef_center_offset, (0.0, 0.0, 0.0)) * desired_base_world
    center_body = _spawn_virtual_body(
        stage,
        prim_path,
        EEF_CENTER_BODY_NAME,
        center_world,
        (0.008, 0.008, 0.008),
        cfg.virtual_link_mass,
        (0.20, 0.80, 0.95),
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/gripper_to_sweep_center_joint",
        gripper_base,
        center_body,
        center_world,
    )
    return robot_root


@configclass
class ShelfForceRobotSpawnerCfg(RigidObjectSpawnerCfg):
    func: Callable = spawn_shelf_force_robot
    ur5e_usd_path: str = UR5E_USD_PATH
    robotiq_usd_path: str = ROBOTIQ_USD_PATH
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = (
        sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        )
    )
    tool_frame_candidates: tuple[str, ...] = _TOOL_FRAME_CANDIDATES
    gripper_base_candidates: tuple[str, ...] = _GRIPPER_BASE_CANDIDATES
    mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rotation_deg: tuple[float, float, float] = (0.0, 90.0, 0.0)
    ft_sensor_size: tuple[float, float, float] = (0.025, 0.025, 0.025)
    ft_sensor_mass: float = 1.0e-3
    virtual_link_mass: float = 1.0e-3
    eef_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.16)


def make_shelf_force_robot_cfg() -> ArticulationCfg:
    """Build the example-style joint-position robot with an inline F/T joint."""
    spawner = ShelfForceRobotSpawnerCfg(
        ur5e_usd_path=UR5E_USD_PATH,
        robotiq_usd_path=ROBOTIQ_USD_PATH,
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawner,
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
                ".*(finger|knuckle).*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                velocity_limit_sim={
                    "shoulder_.*": 3.14,
                    "elbow_joint": 3.14,
                    "wrist_.*": 6.28,
                },
                effort_limit_sim={
                    "shoulder_.*": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_.*": 28.0,
                },
                stiffness=0.0,
                damping=0.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*(finger|knuckle).*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )
