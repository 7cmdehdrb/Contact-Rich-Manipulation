"""Can-specific playback variant of the ConstantVelocity HomeReturn task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import EventCfg
from .env_cfg_constant_velocity_upright_random_size_home import (
    SweepHomeObservationsCfg,
    SweepHomeSceneCfg,
    UR5eOscSweepConstantVelocityUprightRandomSizeHomeEnvCfg,
)

CAN_USD_PATH = "omniverse://192.168.0.13/Library/Shelf/Objects/Can_6/Can_6.usd"
OPEN_TABLE_TOP_HEIGHT = 0.775
CAN_HEIGHT = 0.11913070456885584
CAN_OBSERVATION_Z_OFFSET = 0.5 * CAN_HEIGHT
DEFAULT_CAN_MASS = 0.35


@configclass
class CanSweepHomeSceneCfg(SweepHomeSceneCfg):
    """HomeReturn scene replacing only the target cube with Can_6."""

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CAN_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            # Can_6's rigid root has no MassAPI.  Mass is authored by the
            # prestartup event below, not by UsdFileCfg.mass_props.
            activate_contact_sensors=True,
        ),
        # Can_6's local Z minimum is 0.0, so its root belongs on the tabletop.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.50, 0.0, OPEN_TABLE_TOP_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class CanSweepHomeEventsCfg(EventCfg):
    """Base resets plus explicit root MassAPI authoring for Can_6."""

    set_target_mass = EventTerm(
        func=mdp.define_rigid_object_mass,
        mode="prestartup",
        params={
            "mass": DEFAULT_CAN_MASS,
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class CanSweepHomeObservationsCfg(SweepHomeObservationsCfg):
    """Report the Can center instead of its bottom-origin rigid root."""

    @configclass
    class PolicyCfg(SweepHomeObservationsCfg.PolicyCfg):
        initial_target_pose = ObsTerm(
            func=mdp.initial_target_pose_b_with_z_offset,
            params={
                "command_name": "desired_motion",
                "z_offset": CAN_OBSERVATION_Z_OFFSET,
            },
        )
        current_target_pose = ObsTerm(
            func=mdp.current_target_pose_b_with_z_offset,
            params={
                "z_offset": CAN_OBSERVATION_Z_OFFSET,
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("target_object"),
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR5eOscSweepConstantVelocityUprightRandomSizeHomeCanEnvCfg(
    UR5eOscSweepConstantVelocityUprightRandomSizeHomeEnvCfg
):
    """Checkpoint-compatible HomeReturn playback environment for Can_6."""

    scene: CanSweepHomeSceneCfg = CanSweepHomeSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=False,
    )
    events: CanSweepHomeEventsCfg = CanSweepHomeEventsCfg()
    observations: CanSweepHomeObservationsCfg = CanSweepHomeObservationsCfg()
