"""Independent Sweep v0 configuration; no inheritance from older task EnvCfgs."""
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.stabilize_sweep import StabilizeSweepSceneCfg
from ..mdp.actions.stabilize_sweep import StabilizeSweepActionsCfg, StabilizeSweepOSCActionCfg
from ..mdp.observations.stabilize_sweep import SweepOnlyObservationsCfg
from ..mdp.commands.stabilize_sweep import StabilizeSweepCommandsCfg
from ..mdp.rewards.stabilize_sweep import SweepOnlyTestRewardsCfg, add_action_limit_penalties
from ..mdp.terminations.stabilize_sweep import StabilizeSweepTerminationsCfg
from ..mdp.events.stabilize_sweep import StabilizeSweepTestEventsCfg
from ..mdp.action_functions.wrench_refactor import SweepOnlyClippedOSCAction
from ..mdp.observation_functions.stabilize_sweep import legacy_clipped_previous_action
from ..mdp.reward_functions.stabilize_sweep import linear_reach
from ..mdp.reward_functions.stabilize_sweep import incremental_approach_reach, arm_joint_failure
from ..mdp.reward_functions.sweep_contact_preference import ContactSideSweepProgress, SignedContactSideSweepProgress


@configclass
class SweepEnvCfg(ManagerBasedRLEnvCfg):
    """Independent Sweep family, initialized from the v2 LinearReach contract."""
    scene: StabilizeSweepSceneCfg = StabilizeSweepSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: SweepOnlyObservationsCfg = SweepOnlyObservationsCfg()
    actions: StabilizeSweepActionsCfg = StabilizeSweepActionsCfg()
    commands: StabilizeSweepCommandsCfg = StabilizeSweepCommandsCfg()
    rewards: SweepOnlyTestRewardsCfg = SweepOnlyTestRewardsCfg()
    terminations: StabilizeSweepTerminationsCfg = StabilizeSweepTerminationsCfg()
    events: StabilizeSweepTestEventsCfg = StabilizeSweepTestEventsCfg()
    curriculum: dict = {}
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
    debug_surface_points: bool = True
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

        self.events.object_spawn.params['excluded_x_half_angle'] = 0.0
        self.actions.arm_action.class_type = SweepOnlyClippedOSCAction
        self.observations.policy.actions.func = legacy_clipped_previous_action
        add_action_limit_penalties(self.rewards)
        self.rewards.sweep_progress = self.rewards.sweep_progress.replace(
            func=ContactSideSweepProgress, weight=50.0,
            params={'preferred_multiplier': 1.5, 'other_multiplier': 0.7},
        )
        self.rewards.sweep_retreat.weight = -10.0
        self.rewards.reaching = self.rewards.reaching.replace(
            func=linear_reach, weight=0.8,
            params={'near_radius': 0.10, 'decay_distance': 0.35, 'frame_name': 'ee_frame'},
        )
        self.rewards.obstacle_contact.params['free_force'] = 50.0
        self.rewards.obstacle_contact.weight = -0.04
        self.rewards.wrench_change.params['free_force_change'] = 30.0
        self.rewards.wrench_change.params['free_moment_change'] = 3.0
        self.rewards.wrench_change.weight = -0.1
        # Manager order: physical clone -> noise -> runtime axis clip -> scale.
        # The reward callable reads the unscaled physical sample directly.
        self.observations.policy.wrist_wrench.scale = tuple(
            1.0 / bound for bound in self.wrist_wrench_observation_limits
        )


@configclass
class SweepSingleContactEnvCfg(SweepEnvCfg):
    """Same configuration; the registered runtime selects binary contact credit."""


@configclass
class SweepSingleContactV1EnvCfg(SweepSingleContactEnvCfg):
    """Moving approach progress, signed force alignment and one joint-failure cost."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.sweep_progress = self.rewards.sweep_progress.replace(func=SignedContactSideSweepProgress)
        self.rewards.reaching = self.rewards.reaching.replace(
            func=incremental_approach_reach, weight=4.9, params={},
        )
        self.rewards.arm_joint_failure = RewTerm(func=arm_joint_failure, weight=-3.0)
