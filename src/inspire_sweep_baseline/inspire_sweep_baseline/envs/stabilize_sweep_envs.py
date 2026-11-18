from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.stabilize_sweep import StabilizeSweepSceneCfg
from ..mdp.actions.stabilize_sweep import StabilizeSweepActionsCfg
from ..mdp.observations.stabilize_sweep import StabilizeSweepObservationsCfg
from ..mdp.commands.stabilize_sweep import StabilizeSweepCommandsCfg
from ..mdp.rewards.stabilize_sweep import StabilizeSweepRewardsCfg, SweepOnlyRewardsCfg
from ..mdp.rewards.stabilize_sweep import StabilizeSweepTestRewardsCfg, SweepOnlyTestRewardsCfg
from ..mdp.terminations.stabilize_sweep import StabilizeSweepTerminationsCfg
from ..mdp.events.stabilize_sweep import StabilizeSweepEventsCfg
from ..mdp.events.stabilize_sweep import StabilizeSweepTestEventsCfg
from ..mdp.curricula.stabilize_sweep import StabilizeSweepCurriculumCfg
from ..mdp.actions.stabilize_sweep import StabilizeSweepOSCActionCfg

@configclass
class StabilizeSweepEnvCfg(ManagerBasedRLEnvCfg):
    scene: StabilizeSweepSceneCfg = StabilizeSweepSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: StabilizeSweepObservationsCfg = StabilizeSweepObservationsCfg()
    actions: StabilizeSweepActionsCfg = StabilizeSweepActionsCfg()
    commands: StabilizeSweepCommandsCfg = StabilizeSweepCommandsCfg()
    rewards: StabilizeSweepRewardsCfg = StabilizeSweepRewardsCfg()
    terminations: StabilizeSweepTerminationsCfg = StabilizeSweepTerminationsCfg()
    events: StabilizeSweepEventsCfg = StabilizeSweepEventsCfg()
    curriculum: StabilizeSweepCurriculumCfg = StabilizeSweepCurriculumCfg()
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
        self.actions.arm_action.body_offset = StabilizeSweepOSCActionCfg.OffsetCfg(pos=eef.offset.pos, rot=eef.offset.rot)
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
class SweepOnlyEnvCfg(StabilizeSweepEnvCfg):
    rewards: SweepOnlyRewardsCfg = SweepOnlyRewardsCfg()


@configclass
class StabilizeSweepTestEnvCfg(StabilizeSweepEnvCfg):
    events: StabilizeSweepTestEventsCfg = StabilizeSweepTestEventsCfg()
    rewards: StabilizeSweepTestRewardsCfg = StabilizeSweepTestRewardsCfg()


@configclass
class SweepOnlyTestEnvCfg(SweepOnlyEnvCfg):
    events: StabilizeSweepTestEventsCfg = StabilizeSweepTestEventsCfg()
    rewards: SweepOnlyTestRewardsCfg = SweepOnlyTestRewardsCfg()
