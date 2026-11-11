"""Independent Simple v0; Prof changes goal rewards, v1 changes only Reach."""
from ..mdp.actions.sweep_simple import SweepSimpleFingerV1ActionsCfg
from ..mdp.actions.sweep_simple import SweepSimpleFingerWrenchActionsCfg
from ..mdp.observations.sweep_simple import SweepSimpleFingerWrenchObservationsCfg
from ..mdp.observations.sweep_simple import SweepSimpleFingerV1ObservationsCfg
from ..mdp.rewards.sweep_simple import SweepSimpleFingerV1RewardsCfg, SweepSimpleFingerV2RewardsCfg
from ..mdp.rewards.sweep_simple import SweepSimpleSemiProfRewardsCfg
from ..mdp.rewards.sweep_simple import (
    SweepSimpleFingerRawV1RewardsCfg, SweepSimpleFingerRawV2RewardsCfg,
    SweepSimpleSemiProfRawRewardsCfg,
)
from ..mdp.curricula.sweep_simple import SweepSimpleFingerV1CurriculumCfg
from ..mdp.curricula.sweep_simple import SweepSimpleFingerV0TestCurriculumCfg
from ..mdp.terminations.sweep_simple import SweepSimpleFingerV0TestTerminationsCfg
from ..mdp.terminations.sweep_simple import SweepSimpleFingerV1TerminationsCfg, SweepSimpleFingerV2TerminationsCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.sweep_simple import SweepSimpleSceneCfg
from ..mdp.actions.sweep_simple import SweepSimpleActionsCfg, SweepSimpleOSCActionCfg, SweepSimpleFingerActionsCfg
from ..mdp.commands.sweep_simple import SweepSimpleCommandsCfg
from ..mdp.observations.sweep_simple import SweepSimpleObservationsCfg, SweepSimpleFingerObservationsCfg
from ..mdp.events.sweep_simple import SweepSimpleEventsCfg, SweepSimpleHardEventsCfg
from ..mdp.curricula.sweep_simple import SweepSimpleCurriculumCfg, SweepSimpleHardCurriculumCfg
from ..mdp.terminations.sweep_simple import SweepSimpleTerminationsCfg
from ..mdp.rewards.sweep_simple import SweepSimpleRewardsCfg, SweepSimpleProfRewardsCfg, SweepSimpleV1RewardsCfg, SweepSimpleProfV1RewardsCfg, SweepSimpleFingerRewardsCfg

@configclass
class SweepSimpleEnvCfg(ManagerBasedRLEnvCfg):
    scene: SweepSimpleSceneCfg = SweepSimpleSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: SweepSimpleObservationsCfg = SweepSimpleObservationsCfg()
    actions: SweepSimpleActionsCfg = SweepSimpleActionsCfg()
    commands: SweepSimpleCommandsCfg = SweepSimpleCommandsCfg()
    rewards: SweepSimpleRewardsCfg = SweepSimpleRewardsCfg()
    terminations: SweepSimpleTerminationsCfg = SweepSimpleTerminationsCfg()
    events: SweepSimpleEventsCfg = SweepSimpleEventsCfg()
    curriculum: SweepSimpleCurriculumCfg = SweepSimpleCurriculumCfg()
    obstacle_observation_slots: int = 16
    # Physical roster compatibility only; Fast placement uses max_obstacles.
    max_gray_obstacles: int = 3
    max_white_obstacles: int = 3
    target_bounding_box: bool = False
    goal_success_radius: float = 0.03
    goal_hold_outer_radius: float | None = None
    goal_reward_height_limit: float = 0.14
    debug_initial_ik_bounds: bool = False
    wrist_wrench_observation_limits: tuple = (150.0, 150.0, 150.0, 8.0, 8.0, 8.0)
    red_margin: float = 0.023
    rear_offset: float = 0.02
    contact_height_limit: float = 0.14
    gray_padding: float = 0.065
    viewer: ViewerCfg = ViewerCfg(eye=(3.5, 3.5, 3.5), lookat=(0.0, 0.0, 0.3))

    max_obstacles: int = 6
    placement_clearance: float = 0.01
    layout_cell_size: float = 0.005
    layout_geometry_epsilon: float = 1e-06
    reset_ik_backend: str = 'warp'
    timeout_metric_window: int = 10000

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2 ** 17
        defaults = self.sim.physics_material
        self.scene.robot.spawn.friction_defaults = defaults.copy()
        self.scene.shelf.spawn.friction_defaults = defaults.copy()
        for obj in self.scene.object_collection.rigid_objects.values():
            obj.spawn.friction_defaults = defaults.copy()
        self.scene.configure_population_capacity(self.max_gray_obstacles, self.max_white_obstacles, self.obstacle_observation_slots)
        self.contact_height_limit = self.goal_reward_height_limit
        self.commands.target_goal_pos.goal_radius = self.goal_success_radius
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action.body_name = eef.prim_path.rsplit('/', 1)[-1]
        self.actions.arm_action.body_offset = SweepSimpleOSCActionCfg.OffsetCfg(pos=eef.offset.pos, rot=eef.offset.rot)
        self.scene.robot.actuators['arm'].stiffness = 0.0
        self.scene.robot.actuators['arm'].damping = 0.0
        self.scene.robot.spawn.sensor_mass = 0.28
        self.scene.robot.actuators['hand'].effort_limit_sim = 10.0
        self.scene.robot.actuators['hand'].velocity_limit_sim = 1.0
        self.scene.robot.actuators['arm'].velocity_limit_sim = dict.fromkeys(ARM_JOINT_NAMES, 3.14)
        self.scene.robot.actuators['arm'].effort_limit_sim = dict(zip(ARM_JOINT_NAMES, (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)))
        if self.max_obstacles != 6 or self.reset_ik_backend != 'warp':
            raise ValueError('Fast reset requires max_obstacles=6 and Warp IK')
        self.max_gray_obstacles = self.max_white_obstacles = None
        self.gray_padding = self.red_margin = self.rear_offset = None

@configclass
class SweepSimpleProfEnvCfg(SweepSimpleEnvCfg):
    rewards: SweepSimpleProfRewardsCfg = SweepSimpleProfRewardsCfg()

@configclass
class SweepSimpleV1EnvCfg(SweepSimpleEnvCfg):
    rewards: SweepSimpleV1RewardsCfg = SweepSimpleV1RewardsCfg()

@configclass
class SweepSimpleProfV1EnvCfg(SweepSimpleProfEnvCfg):
    rewards: SweepSimpleProfV1RewardsCfg = SweepSimpleProfV1RewardsCfg()


@configclass
class SweepSimpleHardEnvCfg(SweepSimpleV1EnvCfg):
    events: SweepSimpleHardEventsCfg = SweepSimpleHardEventsCfg()
    curriculum: SweepSimpleHardCurriculumCfg = SweepSimpleHardCurriculumCfg()


@configclass
class SweepSimpleProfHardEnvCfg(SweepSimpleProfV1EnvCfg):
    events: SweepSimpleHardEventsCfg = SweepSimpleHardEventsCfg()
    curriculum: SweepSimpleHardCurriculumCfg = SweepSimpleHardCurriculumCfg()


@configclass
class SweepSimpleFingerEnvCfg(SweepSimpleV1EnvCfg):
    actions: SweepSimpleFingerActionsCfg = SweepSimpleFingerActionsCfg()
    observations: SweepSimpleFingerObservationsCfg = SweepSimpleFingerObservationsCfg()
    rewards: SweepSimpleFingerRewardsCfg = SweepSimpleFingerRewardsCfg()


@configclass
class SweepSimpleFingerWrenchEnvCfg(SweepSimpleFingerEnvCfg):
    actions: SweepSimpleFingerWrenchActionsCfg = SweepSimpleFingerWrenchActionsCfg()
    observations: SweepSimpleFingerWrenchObservationsCfg = SweepSimpleFingerWrenchObservationsCfg()
    terminations: SweepSimpleFingerV0TestTerminationsCfg = SweepSimpleFingerV0TestTerminationsCfg()
    curriculum: SweepSimpleFingerV0TestCurriculumCfg = SweepSimpleFingerV0TestCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Parent aligns pose control with its offset EEF. Wrench control uses
        # the sensor origin; preserve the scene EEF used by observations/rewards.
        self.actions.arm_action.body_name = 'axia80_link'
        self.actions.arm_action.body_offset = None


@configclass
class SweepSimpleFingerV0TestEnvCfg(SweepSimpleFingerEnvCfg):
    terminations: SweepSimpleFingerV0TestTerminationsCfg = SweepSimpleFingerV0TestTerminationsCfg()
    curriculum: SweepSimpleFingerV0TestCurriculumCfg = SweepSimpleFingerV0TestCurriculumCfg()


@configclass
class SweepSimpleFingerV1EnvCfg(SweepSimpleFingerEnvCfg):
    clip_raw_actions: bool = True
    actions: SweepSimpleFingerV1ActionsCfg = SweepSimpleFingerV1ActionsCfg()
    observations: SweepSimpleFingerV1ObservationsCfg = SweepSimpleFingerV1ObservationsCfg()
    rewards: SweepSimpleFingerV1RewardsCfg = SweepSimpleFingerV1RewardsCfg()
    terminations: SweepSimpleFingerV1TerminationsCfg = SweepSimpleFingerV1TerminationsCfg()
    curriculum: SweepSimpleFingerV1CurriculumCfg = SweepSimpleFingerV1CurriculumCfg()


@configclass
class SweepSimpleFingerV2EnvCfg(SweepSimpleFingerV1EnvCfg):
    rewards: SweepSimpleFingerV2RewardsCfg = SweepSimpleFingerV2RewardsCfg()
    terminations: SweepSimpleFingerV2TerminationsCfg = SweepSimpleFingerV2TerminationsCfg()


@configclass
class SweepSimpleSemiProfEnvCfg(SweepSimpleFingerV1EnvCfg):
    rewards: SweepSimpleSemiProfRewardsCfg = SweepSimpleSemiProfRewardsCfg()


@configclass
class SweepSimpleFingerRawV1EnvCfg(SweepSimpleFingerV1EnvCfg):
    clip_raw_actions: bool = False
    actions: SweepSimpleFingerActionsCfg = SweepSimpleFingerActionsCfg()
    observations: SweepSimpleFingerObservationsCfg = SweepSimpleFingerObservationsCfg()
    rewards: SweepSimpleFingerRawV1RewardsCfg = SweepSimpleFingerRawV1RewardsCfg()


@configclass
class SweepSimpleFingerRawV2EnvCfg(SweepSimpleFingerV2EnvCfg):
    clip_raw_actions: bool = False
    actions: SweepSimpleFingerActionsCfg = SweepSimpleFingerActionsCfg()
    observations: SweepSimpleFingerObservationsCfg = SweepSimpleFingerObservationsCfg()
    rewards: SweepSimpleFingerRawV2RewardsCfg = SweepSimpleFingerRawV2RewardsCfg()


@configclass
class SweepSimpleSemiProfRawEnvCfg(SweepSimpleSemiProfEnvCfg):
    clip_raw_actions: bool = False
    actions: SweepSimpleFingerActionsCfg = SweepSimpleFingerActionsCfg()
    observations: SweepSimpleFingerObservationsCfg = SweepSimpleFingerObservationsCfg()
    rewards: SweepSimpleSemiProfRawRewardsCfg = SweepSimpleSemiProfRawRewardsCfg()
