"""Explicit V4-equivalent composition using the existing local MDP functions."""

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
from .omni_env_cfg import OBJECT_RADIUS, SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT
from .realistic_physics_env_cfg import (
    RealisticPhysicsEventsCfg,
    apply_realistic_physics,
)


@configclass
class OmniRenewalEventsCfg(RealisticPhysicsEventsCfg):
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
class OmniRenewalRewardsCfg:
    reaching = RewTerm(
        func=goal_rewards.HeightGatedPlanarReachProgress,
        weight=30.0,
        params={
            "near_radius": 0.10,
            "frame_name": "ee_frame",
            "z_offset": 0.09,
            "z_tolerance": 0.02,
        },
    )
    goal_distance = RewTerm(
        func=goal_rewards.PlanarGoalProgress,
        weight=40.0,
        params={"position_tolerance": 0.01},
    )
    arm_action_rate = RewTerm(
        func=goal_rewards.action_rate_l2,
        weight=-0.03,
        params={"action_group": "arm"},
    )
    hand_action_rate = RewTerm(
        func=goal_rewards.action_rate_l2,
        weight=-0.03,
        params={"action_group": "hand"},
    )
    arm_joint_velocity = RewTerm(
        func=goal_rewards.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": _arm()},
    )
    hand_joint_velocity = RewTerm(
        func=goal_rewards.joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True
            )
        },
    )
    object_tilt = RewTerm(
        func=goal_rewards.object_tilt_cost,
        weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    shelf_collision = RewTerm(func=mdp.ShelfCollisionPenalty, weight=-0.5)
    time = None
    success = None
    failure = None
    pushing_speed = RewTerm(
        func=goal_rewards.pushing_speed_cost,
        weight=-5.0,
        params={"free_speed": 0.2, "reference_speed": 0.4},
    )
    arm_speed_excess = RewTerm(
        func=goal_rewards.joint_speed_cost,
        weight=-0.1,
        params={"asset_cfg": _arm(), "free_speed": 1.0, "reference_speed": 2.0},
    )


@configclass
class OmniRenewalTerminationsCfg(TerminationsCfg):
    object_drop = DoneTerm(
        func=goal_terminations.object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    push_fast = DoneTerm(func=mdp.push_fast, params={"speed_condition": 0.5})
    hand_velocity = DoneTerm(
        func=mdp.arm_velocity, params={"asset_cfg": _arm(), "threshold": 5.0}
    )
    object_outside_shelf = None
    shelf_displacement = DoneTerm(
        func=goal_terminations.shelf_displacement, params={"threshold": 0.05}
    )
    success = None


@configclass
class InspireSweepOmniRenewalEnvCfg(InspireSweepEnvCfg):
    """Compose current V4 behavior directly on the baseline mechanics."""

    events: OmniRenewalEventsCfg = OmniRenewalEventsCfg()
    rewards: OmniRenewalRewardsCfg = OmniRenewalRewardsCfg()
    terminations: OmniRenewalTerminationsCfg = OmniRenewalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_realistic_physics(self)
        self.scene.object_collection.rigid_objects["mass_cylinder"].init_state.pos = (
            -0.7,
            0.0,
            1.05,
        )
