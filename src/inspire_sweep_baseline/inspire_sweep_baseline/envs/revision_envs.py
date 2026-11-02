"""Revision v0: explicit independent Cfg composition from the reviewed v0_2 contract."""

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass

from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.revision import RevisionSceneCfg, RevisionV1TestSceneCfg
from ..mdp.actions.revision import RevisionActionsCfg, RevisionOSCActionCfg
from ..mdp.commands.revision import RevisionCommandsCfg
from ..mdp.common.revision_goal import CONTACT_HEIGHT_LIMIT, GOAL_SUCCESS_RADIUS, HOLD_OUTER_RADIUS
from ..mdp.curricula.revision import RevisionCurriculumCfg, RevisionV1CurriculumCfg
from ..mdp.events.revision import RevisionEventsCfg
from ..mdp.observations.revision import RevisionObservationsCfg
from ..mdp.rewards.revision import RevisionRewardsCfg, RevisionV1RewardsCfg
from ..mdp.terminations.revision import RevisionTerminationsCfg


@configclass
class RevisionEnvCfg(ManagerBasedRLEnvCfg):
    scene: RevisionSceneCfg = RevisionSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: RevisionObservationsCfg = RevisionObservationsCfg()
    actions: RevisionActionsCfg = RevisionActionsCfg()
    commands: RevisionCommandsCfg = RevisionCommandsCfg()
    rewards: RevisionRewardsCfg = RevisionRewardsCfg()
    terminations: RevisionTerminationsCfg = RevisionTerminationsCfg()
    events: RevisionEventsCfg = RevisionEventsCfg()
    curriculum: RevisionCurriculumCfg = RevisionCurriculumCfg()
    obstacle_observation_slots: int = 16
    max_gray_obstacles: int = 4
    max_white_obstacles: int = 5
    target_bounding_box: bool = False
    goal_success_radius: float = GOAL_SUCCESS_RADIUS
    goal_hold_outer_radius: float = HOLD_OUTER_RADIUS
    contact_height_limit: float = CONTACT_HEIGHT_LIMIT
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
            self.max_gray_obstacles, self.max_white_obstacles, self.obstacle_observation_slots,
        )
        self.events.object_spawn.params.update(gray_obstacles=1, white_obstacles=1, gray_padding=self.gray_padding)
        self.rewards.target_goal_hold.params.update(
            position_tolerance=self.goal_success_radius, outer_radius=self.goal_hold_outer_radius,
        )
        self.commands.target_goal_pos.goal_radius = self.goal_success_radius
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action.body_name = eef.prim_path.rsplit("/", 1)[-1]
        self.actions.arm_action.body_offset = RevisionOSCActionCfg.OffsetCfg(pos=eef.offset.pos, rot=eef.offset.rot)
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0


@configclass
class RevisionV1EnvCfg(RevisionEnvCfg):
    rewards: RevisionV1RewardsCfg = RevisionV1RewardsCfg()
    curriculum: RevisionV1CurriculumCfg = RevisionV1CurriculumCfg()
    # Compatibility field only; v1 has no outer Hold tail.
    goal_hold_outer_radius: float | None = None

    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params.update(gray_obstacles=3, white_obstacles=3)
        self.rewards.target_goal_hold.params = {
            "position_tolerance": self.goal_success_radius, "free_speed": 0.05, "speed_scale": 0.05,
        }


@configclass
class RevisionV1TestEnvCfg(RevisionV1EnvCfg):
    scene: RevisionV1TestSceneCfg = RevisionV1TestSceneCfg(num_envs=2048, env_spacing=2.5)
