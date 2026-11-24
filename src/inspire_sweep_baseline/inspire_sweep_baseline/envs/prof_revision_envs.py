"""Independent Prof Revision v0 configuration."""
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.prof_revision import ProfRevisionSceneCfg
from ..mdp.actions.prof_revision import ProfRevisionActionsCfg, ProfRevisionOSCActionCfg
from ..mdp.observations.prof_revision import ProfRevisionObservationsCfg
from ..mdp.commands.prof_revision import ProfRevisionCommandsCfg
from ..mdp.rewards.prof_revision import ProfRevisionRewardsCfg, ProfRevisionV1RewardsCfg, ProfRevisionV0_1RewardsCfg
from ..mdp.terminations.prof_revision import ProfRevisionTerminationsCfg
from ..mdp.events.prof_revision import ProfRevisionEventsCfg
from ..mdp.action_functions.wrench_refactor import SweepOnlyClippedOSCAction
from ..mdp.action_functions.sweep_simple import NormalizedFingerJointPositionAction
from ..mdp.observation_functions.stabilize_sweep import legacy_clipped_previous_action


@configclass
class ProfRevisionEnvCfg(ManagerBasedRLEnvCfg):
    """Initial observed OBB and tactile sweep, without environment inheritance."""
    scene: ProfRevisionSceneCfg = ProfRevisionSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ProfRevisionObservationsCfg = ProfRevisionObservationsCfg()
    actions: ProfRevisionActionsCfg = ProfRevisionActionsCfg()
    commands: ProfRevisionCommandsCfg = ProfRevisionCommandsCfg()
    rewards: ProfRevisionRewardsCfg = ProfRevisionRewardsCfg()
    terminations: ProfRevisionTerminationsCfg = ProfRevisionTerminationsCfg()
    events: ProfRevisionEventsCfg = ProfRevisionEventsCfg()
    curriculum: dict = {}
    obstacle_observation_slots: int = 16
    # Physical roster compatibility only; Fast placement uses max_obstacles.
    max_gray_obstacles: int = 3
    max_white_obstacles: int = 3
    target_bounding_box: bool = False
    goal_success_radius: float = 0.03
    goal_hold_outer_radius: float | None = None
    debug_initial_ik_bounds: bool = False
    debug_surface_points: bool = False
    obb_minimum_width: float = 0.02
    obb_minimum_ratio: float = 0.20
    reach_near_radius: float = 0.10
    formation_distance_scale: float = 0.05
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
        import math
        for name in ('obb_minimum_width', 'formation_distance_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.obb_minimum_ratio) or not 0. <= self.obb_minimum_ratio <= 1.:
            raise ValueError('obb_minimum_ratio must be in [0,1]')
        if not math.isfinite(self.reach_near_radius) or self.reach_near_radius < 0.:
            raise ValueError('reach_near_radius must be finite and nonnegative')
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
        self.actions.arm_action.body_offset = ProfRevisionOSCActionCfg.OffsetCfg(pos=eef.offset.pos, rot=eef.offset.rot)
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
        self.actions.gripper_action.class_type = NormalizedFingerJointPositionAction
        self.observations.policy.actions.func = legacy_clipped_previous_action
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
class ProfRevisionReach5cmEnvCfg(ProfRevisionEnvCfg):
    """Parent behavior with a five-centimeter Reaching tolerance."""

    reach_near_radius: float = 0.05


@configclass
class ProfRevisionV1EnvCfg(ProfRevisionEnvCfg):
    """Two-sided finger formation and a one-off arm speed termination cost."""

    rewards: ProfRevisionV1RewardsCfg = ProfRevisionV1RewardsCfg()
    formation_mode: str = 'distance_angle_sum'


@configclass
class ProfRevisionV0_1EnvCfg(ProfRevisionEnvCfg):
    """Parent v0 with only the one-off arm speed termination cost added."""

    rewards: ProfRevisionV0_1RewardsCfg = ProfRevisionV0_1RewardsCfg()
