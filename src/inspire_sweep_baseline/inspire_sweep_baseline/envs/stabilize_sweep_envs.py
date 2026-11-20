from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.stabilize_sweep import StabilizeSweepSceneCfg
from ..mdp.actions.stabilize_sweep import StabilizeSweepActionsCfg
from ..mdp.observations.stabilize_sweep import StabilizeSweepObservationsCfg, SweepOnlyObservationsCfg
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
    """Shared contract; children select stabilization, reset style and Reach."""
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
    # The actual Sweep contact gate is relative to the shelf surface.
    sweep_contact_height_limit: float = 0.12
    sweep_normalize_contact_count: bool = True
    debug_initial_ik_bounds: bool = False
    debug_surface_points: bool = False
    wrist_wrench_observation_limits: tuple = (150.0, 150.0, 150.0, 8.0, 8.0, 8.0)
    red_margin: float = 0.023
    rear_offset: float = 0.02
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
    observations: SweepOnlyObservationsCfg = SweepOnlyObservationsCfg()
    rewards: SweepOnlyRewardsCfg = SweepOnlyRewardsCfg()


@configclass
class StabilizeSweepTestEnvCfg(StabilizeSweepEnvCfg):
    debug_surface_points: bool = True
    events: StabilizeSweepTestEventsCfg = StabilizeSweepTestEventsCfg()
    rewards: StabilizeSweepTestRewardsCfg = StabilizeSweepTestRewardsCfg()


@configclass
class SweepOnlyTestEnvCfg(SweepOnlyEnvCfg):
    debug_surface_points: bool = True
    events: StabilizeSweepTestEventsCfg = StabilizeSweepTestEventsCfg()
    rewards: SweepOnlyTestRewardsCfg = SweepOnlyTestRewardsCfg()


@configclass
class SweepOnlyV1TestEnvCfg(SweepOnlyTestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.sweep_progress.weight = 50.0
        self.rewards.sweep_retreat.weight = -10.0


@configclass
class SweepOnlyV1_2TestEnvCfg(SweepOnlyV1TestEnvCfg):
    sweep_normalize_contact_count: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.rewards.sweep_progress.weight = 100.0
        self.rewards.sweep_retreat.weight = -25.0


@configclass
class SweepOnlyV2TestEnvCfg(SweepOnlyV1TestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params['excluded_x_half_angle'] = 0.0


@configclass
class SweepOnlyV2_2TestEnvCfg(SweepOnlyV1_2TestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params['excluded_x_half_angle'] = 0.0


def _configure_action_penalty(cfg):
    """Only children opt in; the runner must pass unclipped commands to the manager."""
    from ..mdp.action_functions.wrench_refactor import SweepOnlyClippedOSCAction
    from ..mdp.observation_functions.stabilize_sweep import legacy_clipped_previous_action
    from ..mdp.rewards.stabilize_sweep import add_action_limit_penalties
    cfg.actions.arm_action.class_type = SweepOnlyClippedOSCAction
    # FingerJointPositionAction already limits targets to [0, URDF upper].
    # Rate penalties still read manager action/prev_action, now BEFORE clipping.
    cfg.observations.policy.actions.func = legacy_clipped_previous_action
    add_action_limit_penalties(cfg.rewards)


@configclass
class SweepOnlyActionPenaltyEnvCfg(SweepOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_action_penalty(self)


@configclass
class SweepOnlyTestActionPenaltyEnvCfg(SweepOnlyTestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_action_penalty(self)


@configclass
class SweepOnlyV1TestActionPenaltyEnvCfg(SweepOnlyV1TestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_action_penalty(self)


@configclass
class SweepOnlyV1_2TestActionPenaltyEnvCfg(SweepOnlyV1_2TestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_action_penalty(self)


@configclass
class SweepOnlyV2TestActionPenaltyEnvCfg(SweepOnlyV2TestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_action_penalty(self)


@configclass
class SweepOnlyV2_2TestActionPenaltyEnvCfg(SweepOnlyV2_2TestEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_action_penalty(self)


@configclass
class SweepOnlyV1ActionPenaltyEnvCfg(SweepOnlyEnvCfg):
    """Adjacent-object IK from v0, v1 Sweep weights and raw-action penalties."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.sweep_progress.weight = 50.0
        self.rewards.sweep_retreat.weight = -10.0
        _configure_action_penalty(self)


@configclass
class SweepOnlyV2TestActionPenaltyLinearReachEnvCfg(SweepOnlyV2TestActionPenaltyEnvCfg):
    """Preserve v2-Test-ActionPenalty; replace only its continuous Reach."""

    def __post_init__(self):
        super().__post_init__()
        from ..mdp.reward_functions.stabilize_sweep import linear_reach
        self.rewards.reaching = self.rewards.reaching.replace(
            func=linear_reach, weight=0.8,
            params={'near_radius': 0.10, 'decay_distance': 0.35, 'frame_name': 'ee_frame'},
        )
