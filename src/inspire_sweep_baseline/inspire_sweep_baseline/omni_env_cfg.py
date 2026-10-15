"""Baseline mechanics with realistic physics and goal-based omnidirectional rewards."""

import math

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from . import mdp
from .assets.robot import HAND_JOINT_NAMES
from .env_cfg import InspireSweepEnvCfg, TerminationsCfg, _arm
from .mdp import goal_rewards, goal_terminations
from .mdp.goal_events import randomize_goal_object_scene
from .realistic_physics_env_cfg import (
    RealisticPhysicsEventsCfg,
    apply_realistic_physics,
)

# Measured middle-board dimensions of the same shelf asset used by Baseline.
# Shelf-local XY is horizontal and Z is up; the shelf root is (-0.7, 0, 0).
SHELF_HALF_EXTENTS = (0.20, 0.50)
SHELF_SURFACE_HEIGHT = 1.05
OBJECT_RADIUS = 0.04
FAILURE_TERMS = (
    "object_drop",
    "push_fast",
    "shelf_collision",
    "hand_velocity",
    "object_outside_shelf",
    "shelf_displacement",
)


@configclass
class OmniEventsCfg(RealisticPhysicsEventsCfg):
    object_spawn = EventTerm(
        func=randomize_goal_object_scene,
        mode="reset",
        params={
            "spawn_radius": 0.02,
            "goal_distance": 0.15,
            "object_radius": OBJECT_RADIUS,
            "edge_clearance": 0.02,
            "shelf_half_extents": SHELF_HALF_EXTENTS,
            "shelf_surface_height": SHELF_SURFACE_HEIGHT,
        },
    )


@configclass
class OmniRewardsCfg:
    # Dense costs are per second: RewardManager applies step_dt (T_ref = 1 s).
    goal_distance = RewTerm(
        func=goal_rewards.planar_goal_cost, weight=6.0, params={"sigma": 0.10}
    )
    reaching = RewTerm(
        func=goal_rewards.ReachProgress,
        weight=1.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )

    arm_action_rate = RewTerm(
        func=goal_rewards.normalized_action_rate,
        weight=-0.03,
        params={"action_group": "arm", "normalization_scale": 1.0, "reduction": "sum"},
    )
    hand_action_rate = RewTerm(
        func=goal_rewards.normalized_action_rate,
        weight=-0.03,
        params={"action_group": "hand", "normalization_scale": 1.0},
    )
    arm_joint_velocity = RewTerm(
        func=goal_rewards.normalized_joint_velocity,
        weight=-0.03,
        params={"asset_cfg": _arm(), "reference_velocity": 1.0, "reduction": "sum"},
    )
    hand_joint_velocity = RewTerm(
        func=goal_rewards.normalized_joint_velocity,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            ),
            "reference_velocity": 1.0,
        },
    )

    object_tilt = RewTerm(
        func=goal_rewards.object_tilt_cost,
        weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    shelf_collision = RewTerm(func=mdp.ShelfCollisionPenalty, weight=-0.5)

    time = RewTerm(func=goal_rewards.time_cost, weight=-0.1)
    # Progress and terminal functions undo the manager's dt multiplication.
    success = RewTerm(func=goal_rewards.success_event, weight=30.0)
    failure = RewTerm(
        func=goal_rewards.failure_event,
        weight=-80.0,
        params={"failure_terms": FAILURE_TERMS},
    )


@configclass
class OmniTerminationsCfg(TerminationsCfg):
    object_drop = DoneTerm(
        func=goal_terminations.object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    object_outside_shelf = DoneTerm(
        func=goal_terminations.object_outside_shelf,
        params={"half_extents": SHELF_HALF_EXTENTS},
    )
    shelf_displacement = DoneTerm(
        func=goal_terminations.shelf_displacement, params={"threshold": 0.05}
    )
    # Keep this after all failure terms: the hold counter reads their current results.
    success = DoneTerm(
        func=goal_terminations.StableGoalSuccess,
        params={
            "position_tolerance": 0.01,
            "speed_tolerance": 0.02,
            "upright_angle": math.radians(10.0),
            "hold_time": 0.3,
            "surface_height": SHELF_SURFACE_HEIGHT,
            "support_tolerance": 0.01,
            "half_extents": SHELF_HALF_EXTENTS,
            "object_radius": OBJECT_RADIUS,
            "failure_terms": FAILURE_TERMS,
        },
    )


@configclass
class InspireSweepOmniEnvCfg(InspireSweepEnvCfg):
    events: OmniEventsCfg = OmniEventsCfg()
    rewards: OmniRewardsCfg = OmniRewardsCfg()
    terminations: OmniTerminationsCfg = OmniTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_realistic_physics(self)
        # Baseline policy remains the live-object 41D observation; actions,
        # scene assets, initial joints, commands, timing and PPO are inherited.
        self.scene.object_collection.rigid_objects["mass_cylinder"].init_state.pos = (
            -0.7,
            0.0,
            1.05,
        )


@configclass
class InspireSweepOmniV1EnvCfg(InspireSweepOmniEnvCfg):
    """Omni variant with a 100 rad/s arm termination threshold."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.hand_velocity.params["threshold"] = 3.0
