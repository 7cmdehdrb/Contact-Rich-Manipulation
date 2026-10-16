"""Fixed palm-up Inspire hand made from the production tactile task's USD.

Import this module after :class:`isaaclab.app.AppLauncher` starts Isaac Sim.
The source asset is referenced read-only. All edits are authored in the live
stage, retaining original meshes and inertias. Exposed test pads are moved
and their fixed attachments re-anchored; the source USD is never saved.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from inspire_sweep_baseline.assets.robot import (
    ARM_BODY_NAMES,
    HAND_BASE_BODY_NAME,
    HAND_JOINT_NAMES,
    HAND_LINK_BODY_NAMES,
    HAND_OPEN_TARGETS,
    PALM_SENSOR_BODY_NAMES,
    ROBOT_USD_PATH,
    _matrix_to_pose,
    _set_world_transform,
    _offset_matrix,
    _move_hand_rigid_bodies,
    _world_transform,
    make_robot_cfg,
)


@lru_cache(maxsize=1)
def palm_up_orientation() -> tuple[float, float, float, float]:
    """Rotate the actual palm pad's outward normal onto world +Z.

    The source hand's palm normal is approximately hand-base +Y, not +Z.
    Deriving this from the unchanged USD also retains its small authored tilt.
    """
    stage = Usd.Stage.Open(str(ROBOT_USD_PATH))
    if stage is None:
        raise RuntimeError(f"Cannot open production hand source: {ROBOT_USD_PATH}")
    root = stage.GetDefaultPrim().GetPath()
    base = stage.GetPrimAtPath(root.AppendChild(HAND_BASE_BODY_NAME))
    palm = stage.GetPrimAtPath(root.AppendChild(PALM_SENSOR_BODY_NAMES[0]))
    relative = _world_transform(palm) * _world_transform(base).GetInverse()
    normal = relative.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized()
    quaternion = Gf.Rotation(normal, Gf.Vec3d(0.0, 0.0, 1.0)).GetQuat().GetNormalized()
    return (float(quaternion.GetReal()), *(float(value) for value in quaternion.GetImaginary()))


def _prepare_hand_stage(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    """Strip the arm from a live source instance and fix its hand base.

    The production hand and pad bodies are siblings, so their whole subtree
    must undergo the same rigid transform; changing only the hand-base prim
    would leave detached pad and finger frames at the original arm pose.
    """
    root = stage.GetPrimAtPath(prim_path)
    expected_bodies = {*HAND_LINK_BODY_NAMES, *PALM_SENSOR_BODY_NAMES}
    actual_bodies = {
        prim.GetName()
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName().startswith("inspire_")
    }
    if actual_bodies != expected_bodies:
        raise RuntimeError(
            "Production hand topology changed: "
            f"missing={sorted(expected_bodies - actual_bodies)}, "
            f"extra={sorted(actual_bodies - expected_bodies)}"
        )

    # Gather before mutation because disabling prims changes stage traversal.
    old_joints = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)]
    for prim in old_joints:
        joint = UsdPhysics.Joint(prim)
        bodies = [*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets()]
        if not bodies or any(body.name not in expected_bodies for body in bodies):
            prim.SetActive(False)

    for name in ARM_BODY_NAMES:
        prim = stage.GetPrimAtPath(f"{prim_path}/{name}")
        if prim.IsValid():
            prim.SetActive(False)

    base = _move_hand_rigid_bodies(stage, prim_path, _world_transform(root))
    world_position, world_quaternion = _matrix_to_pose(_world_transform(base))
    # The old root joint was attached to the UR5e base and carries the original
    # articulation root API. Create a fresh root for the remaining hand only.
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{prim_path}/hand_root_joint"))
    joint.CreateBody0Rel().SetTargets([])
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_position))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(world_quaternion[0], Gf.Vec3f(*world_quaternion[1:])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    return root


def _expose_pads(stage, prim_path, clearance):
    """Expose original pad meshes without altering the source USD or parent hulls.

    Translate each sensor body in its own +Z until its lowest mesh vertex is
    above the parent's greatest +Z vertex, then re-anchor its fixed joint.
    Sensor-local meshes/footprints stay unchanged; only this test fixture moves.
    """
    from .targets import load_regions
    joints = [UsdPhysics.Joint(p) for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path))
              if p.IsA(UsdPhysics.FixedJoint)]
    for region in load_regions():
        sensor = stage.GetPrimAtPath(f"{prim_path}/{region['name']}")
        parent_top = region["outward_z_max_m"] - 0.0005
        lift = max(0.0, parent_top + clearance - region["pad_bounds_m"]["min"][2])
        desired_world = _offset_matrix((0.0, 0.0, lift)) * _world_transform(sensor)
        _set_world_transform(sensor, desired_world)
        sensor.CreateAttribute("validation:padLiftM", Sdf.ValueTypeNames.Double).Set(lift)
        matches = [j for j in joints if j.GetBody1Rel().GetTargets() == [sensor.GetPath()]]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one fixed pad attachment for {sensor.GetPath()}")
        joint = matches[0]
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        position, rotation = _matrix_to_pose(desired_world * _world_transform(parent).GetInverse())
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(rotation[0], Gf.Vec3f(*rotation[1:])))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


@clone
def spawn_fixed_inspire_hand(
    prim_path: str,
    cfg: "FixedInspireHandSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Reference production geometry, then remove the arm before cloning."""
    del kwargs
    source_cfg = sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=cfg.rigid_props,
    )
    source_cfg.func(prim_path, source_cfg, translation=translation, orientation=orientation)
    stage = get_current_stage()
    root = _prepare_hand_stage(stage, prim_path)
    if cfg.expose_pads:
        _expose_pads(stage, prim_path, cfg.pad_clearance)
    sim_utils.modify_articulation_root_properties(prim_path, cfg.articulation_props, stage=stage)
    return root


@configclass
class FixedInspireHandSpawnerCfg(RigidObjectSpawnerCfg):
    """World-fixed hand; optional test-only exposure of original pad meshes."""

    func: Callable = spawn_fixed_inspire_hand
    expose_pads: bool = False
    pad_clearance: float = 0.001
    activate_contact_sensors: bool = True
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        # Gravity-free fingers keep the open test pose from sagging before a
        # trial. Probe loading still acts through the original hand dynamics.
        disable_gravity=True,
        max_depenetration_velocity=5.0,
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        sleep_threshold=0.0,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=1,
    )


def make_hand_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot", root_height: float = 0.35,
    *, pad_layout="original", pad_clearance=0.001,
) -> ArticulationCfg:
    """Return a fixed palm-up hand at its production fully open joint targets.

    Hand PD gains and effort/velocity limits are copied from the active tactile
    environment, so the test does not silently replace its sensing geometry or
    finger mechanics. The root is fixed; finger articulation remains dynamic.
    """
    if pad_layout not in ("original", "exposed"):
        raise ValueError("pad_layout must be original or exposed")
    if not math.isfinite(pad_clearance) or pad_clearance <= 0:
        raise ValueError("pad_clearance must be positive and finite")
    if not math.isfinite(root_height):
        raise ValueError("root_height must be finite")
    production_hand_actuator = make_robot_cfg().actuators["hand"].copy()
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=FixedInspireHandSpawnerCfg(expose_pads=pad_layout == "exposed", pad_clearance=pad_clearance),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, root_height),
            rot=palm_up_orientation(),
            joint_pos=dict(HAND_OPEN_TARGETS),
            joint_vel={name: 0.0 for name in HAND_JOINT_NAMES},
        ),
        actuators={"hand": production_hand_actuator},
    )


__all__ = ["FixedInspireHandSpawnerCfg", "make_hand_cfg", "palm_up_orientation", "spawn_fixed_inspire_hand"]
