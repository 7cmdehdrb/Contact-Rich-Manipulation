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

    # 물체 근처로 이동하는 Reaching 보상
    reaching = RewTerm(
        func=goal_rewards.ReachProgress,
        weight=1.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    # 물체를 목표 위치로 이동시키는 Goal Distance 보상
    goal_distance = RewTerm(
        func=goal_rewards.planar_goal_cost, weight=6.0, params={"sigma": 0.10}
    )

    # Mandatory Inspire-compatible squared sums; keep Arm/Hand costs separate.
    # Raw action differences (not action magnitudes or clipped Hand commands).
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
            ),
        },
    )

    # 물체가 떨어지거나 기울어지는 것을 방지하는 보상
    object_tilt = RewTerm(
        func=goal_rewards.object_tilt_cost,
        weight=-2.0,
        params={"free_angle": math.radians(5.0), "fail_angle": 0.9},
    )
    # 물체가 선반 밖으로 나가는 것을 방지하는 보상
    shelf_collision = RewTerm(func=mdp.ShelfCollisionPenalty, weight=-0.5)

    # 물체를 목표 위치로 이동시키는 것을 성공으로 간주하는 보상
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
    """Omni variant with a 3 rad/s arm termination threshold."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.hand_velocity.params["threshold"] = 1.5


@configclass
class OmniV2RewardsCfg(OmniRewardsCfg):
    # Experimental budgets per episode, independent of dt and episode length.
    # Before contact, reaching provides the main positive learning signal.
    # After approach is exhausted, only object-goal progress pays task reward.
    reaching = RewTerm(
        func=goal_rewards.CurrentObjectReachProgress,
        weight=30.0,
        params={"near_radius": 0.10, "frame_name": "ee_frame"},
    )
    goal_distance = RewTerm(
        func=goal_rewards.PlanarGoalProgress,
        weight=40.0,
        params={"position_tolerance": 0.01},
    )
    time = None
    success = None
    failure = None


@configclass
class InspireSweepOmniV2EnvCfg(InspireSweepOmniV1EnvCfg):
    """Live-object approach and sweep progress without success termination."""

    rewards: OmniV2RewardsCfg = OmniV2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success = None


@configclass
class InspireSweepOmniV3EnvCfg(InspireSweepOmniV2EnvCfg):
    """V2 with height-gated XY reaching; all other settings are inherited."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.reaching.func = goal_rewards.HeightGatedPlanarReachProgress
        self.rewards.reaching.params.update(z_offset=0.09, z_tolerance=0.02)


@configclass
class OmniV4RewardsCfg(OmniV2RewardsCfg):
    pushing_speed = RewTerm(
        func=goal_rewards.pushing_speed_cost,
        weight=-0.05,
        params={"free_speed": 0.3, "reference_speed": 0.5},
    )
    arm_speed_excess = RewTerm(
        func=goal_rewards.joint_speed_cost,
        weight=-0.03,
        params={"asset_cfg": _arm(), "free_speed": 1.0, "reference_speed": 2.0},
    )


@configclass
class InspireSweepOmniV4EnvCfg(InspireSweepOmniV3EnvCfg):
    """V3 with relaxed failure limits and object/Arm excess-speed costs."""

    rewards: OmniV4RewardsCfg = OmniV4RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the saved V3 run 2026-10-15_16-53-58_height_gated_xy_reach.
        self.terminations.hand_velocity.params["threshold"] = 5.0
        self.terminations.object_outside_shelf = None
        self.terminations.success = None
        self.terminations.push_fast.params["speed_condition"] = 0.5
