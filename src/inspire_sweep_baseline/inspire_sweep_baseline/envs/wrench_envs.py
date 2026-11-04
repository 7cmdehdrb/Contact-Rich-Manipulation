"""Wrench-v0 directly composes the Test-v2 contract without environment inheritance."""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.wrench import WrenchSceneCfg
from ..mdp.actions.wrench import (
    WrenchActionsCfg,
    WrenchOSCActionCfg,
    WrenchV1ActionsCfg,
)
from ..mdp.commands.wrench import WrenchCommandsCfg
from ..mdp.curricula.wrench import WrenchCurriculumCfg, WrenchV0_1CurriculumCfg
from ..mdp.events.wrench import WrenchEventsCfg
from ..mdp.observations.wrench import WrenchObservationsCfg
from ..mdp.rewards.wrench import WrenchRewardsCfg, WrenchV0_1RewardsCfg, WrenchV1RewardsCfg
from ..mdp.terminations.wrench import WrenchTerminationsCfg


@configclass
class WrenchEnvCfg(ManagerBasedRLEnvCfg):
    scene: WrenchSceneCfg = WrenchSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: WrenchObservationsCfg = WrenchObservationsCfg()
    actions: WrenchActionsCfg = WrenchActionsCfg()
    commands: WrenchCommandsCfg = WrenchCommandsCfg()
    rewards: WrenchRewardsCfg = WrenchRewardsCfg()
    terminations: WrenchTerminationsCfg = WrenchTerminationsCfg()
    events: WrenchEventsCfg = WrenchEventsCfg()
    curriculum: WrenchCurriculumCfg = WrenchCurriculumCfg()
    obstacle_observation_slots: int = 16
    max_gray_obstacles: int = 4
    max_white_obstacles: int = 5
    target_bounding_box: bool = False
    goal_success_radius: float = 0.03
    goal_hold_outer_radius: float | None = None
    goal_reward_height_limit: float = 0.12
    layout_bank_size: int = 16
    layout_build_batch_size: int = 256
    red_margin: float = 0.023
    rear_offset: float = 0.02
    contact_height_limit: float = 0.12
    gray_padding: float = 0.065
    viewer: ViewerCfg = ViewerCfg(eye=(3.5, 3.5, 3.5), lookat=(0.0, 0.0, 0.3))

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        # Use Isaac Lab's default budget instead of the old 8x reservation;
        # validate simultaneous 2048-env resets and contact rollout for overflow.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.scene.robot.actuators["arm"].velocity_limit_sim = dict(
            zip(ARM_JOINT_NAMES, (2.0, 2.0, 2.0, 4.0, 4.0, 4.0))
        )
        defaults = self.sim.physics_material
        self.scene.robot.spawn.friction_defaults = defaults.copy()
        self.scene.shelf.spawn.friction_defaults = defaults.copy()
        for obj in self.scene.object_collection.rigid_objects.values():
            obj.spawn.friction_defaults = defaults.copy()
        self.scene.configure_population_capacity(
            self.max_gray_obstacles,
            self.max_white_obstacles,
            self.obstacle_observation_slots,
        )
        self.events.object_spawn.params.update(
            gray_obstacles=3, white_obstacles=3, gray_padding=self.gray_padding
        )
        self.contact_height_limit = self.goal_reward_height_limit
        self.rewards.target_goal_hold.params = {
            "full_reward_radius": 0.025,
            "outer_radius": 0.035,
        }
        self.commands.target_goal_pos.goal_radius = self.goal_success_radius
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action.body_name = eef.prim_path.rsplit("/", 1)[-1]
        self.actions.arm_action.body_offset = WrenchOSCActionCfg.OffsetCfg(
            pos=eef.offset.pos, rot=eef.offset.rot
        )
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0


@configclass
class WrenchV0_1EnvCfg(WrenchEnvCfg):
    """Contact/angle feasibility variant with a fixed 40-degree goal deadzone."""

    rewards: WrenchV0_1RewardsCfg = WrenchV0_1RewardsCfg()
    curriculum: WrenchV0_1CurriculumCfg = WrenchV0_1CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params.update(
            excluded_x_half_angle=math.radians(40.0), goal_distance=0.075,
        )
        self.events.layout_bank.params.update(
            stages=((40, 3, 3), (40, 3, 4), (40, 4, 4), (40, 4, 5)),
            min_goal_distance=0.025,
        )


@configclass
class WrenchV1EnvCfg(WrenchEnvCfg):
    """Sensor-frame wrench OSC with normalized desired-wrench action-rate cost."""

    actions: WrenchV1ActionsCfg = WrenchV1ActionsCfg()
    rewards: WrenchV1RewardsCfg = WrenchV1RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Parent config installs the EEF offset. Only the control point changes;
        # ee_frame stays unchanged for observations, height rewards and telemetry.
        self.actions.arm_action.body_name = "axia80_link"
        self.actions.arm_action.body_offset = None


@configclass
class WrenchV2EnvCfg(WrenchV1EnvCfg):
    """One tenth of v1's desired and corrected Arm wrench ranges.

    The inherited action-rate cost normalizes by this policy target range;
    its physical scale cancels, so the same relative change keeps its cost.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.wrench_scale = tuple(
            value / 10.0 for value in self.actions.arm_action.wrench_scale
        )
        self.actions.arm_action.commanded_wrench_limit = tuple(
            value / 10.0 for value in self.actions.arm_action.commanded_wrench_limit
        )
