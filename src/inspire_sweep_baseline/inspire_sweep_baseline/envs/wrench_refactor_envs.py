"""Independent Wrench Refactor v0; no previous environment CFG inheritance."""
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.wrench_refactor import WrenchRefactorSceneCfg
from ..mdp.actions.wrench_refactor import WrenchRefactorActionsCfg, WrenchRefactorOSCActionCfg
from ..mdp.observations.wrench_refactor import WrenchRefactorObservationsCfg, WrenchRefactorRelativeObservationsCfg
from ..mdp.commands.wrench_refactor import WrenchRefactorCommandsCfg
from ..mdp.rewards.wrench_refactor import WrenchRefactorRewardsCfg, WrenchRefactorV1RewardsCfg, WrenchRefactorV2RewardsCfg
from ..mdp.terminations.wrench_refactor import WrenchRefactorTerminationsCfg
from ..mdp.events.wrench_refactor import WrenchRefactorEventsCfg
from ..mdp.curricula.wrench_refactor import WrenchRefactorCurriculumCfg


@configclass
class WrenchRefactorEnvCfg(ManagerBasedRLEnvCfg):
    scene: WrenchRefactorSceneCfg = WrenchRefactorSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: WrenchRefactorObservationsCfg = WrenchRefactorObservationsCfg()
    actions: WrenchRefactorActionsCfg = WrenchRefactorActionsCfg()
    commands: WrenchRefactorCommandsCfg = WrenchRefactorCommandsCfg()
    rewards: WrenchRefactorRewardsCfg = WrenchRefactorRewardsCfg()
    terminations: WrenchRefactorTerminationsCfg = WrenchRefactorTerminationsCfg()
    events: WrenchRefactorEventsCfg = WrenchRefactorEventsCfg()
    curriculum: WrenchRefactorCurriculumCfg = WrenchRefactorCurriculumCfg()
    obstacle_observation_slots: int = 16
    max_gray_obstacles: int = 3
    max_white_obstacles: int = 3
    target_bounding_box: bool = False
    goal_success_radius: float = 0.03
    goal_hold_outer_radius: float | None = None
    goal_reward_height_limit: float = 0.14
    red_margin: float = 0.023
    rear_offset: float = 0.02
    contact_height_limit: float = 0.14
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
        self.contact_height_limit = self.goal_reward_height_limit
        self.rewards.target_goal_hold.params = {
            "full_reward_radius": 0.025,
            "outer_radius": 0.035,
        }
        self.commands.target_goal_pos.goal_radius = self.goal_success_radius
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action.body_name = eef.prim_path.rsplit("/", 1)[-1]
        self.actions.arm_action.body_offset = WrenchRefactorOSCActionCfg.OffsetCfg(
            pos=eef.offset.pos, rot=eef.offset.rot
        )
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0



@configclass
class WrenchRefactorV1EnvCfg(WrenchRefactorEnvCfg):
    """Direct v0 child; change only contact-maintenance and goal-distance rewards."""
    rewards: WrenchRefactorV1RewardsCfg = WrenchRefactorV1RewardsCfg()


@configclass
class WrenchRefactorV2EnvCfg(WrenchRefactorEnvCfg):
    """Direct v0 child with signed cumulative-displacement distance reward."""
    rewards: WrenchRefactorV2RewardsCfg = WrenchRefactorV2RewardsCfg()


@configclass
class WrenchRefactorV3EnvCfg(WrenchRefactorV2EnvCfg):
    """Preserve v2 rewards; restore the offset EEF task frame and kinematics."""
    def __post_init__(self):
        super().__post_init__()
        from ..mdp.action_functions.wrench_refactor import WrenchRefactorOSCAction
        self.actions.arm_action.class_type = WrenchRefactorOSCAction


@configclass
class WrenchRefactorV3TestEnvCfg(WrenchRefactorV3EnvCfg):
    """Temporary v3 diagnostic: sample only leftward (shelf -Y) goals."""
    def __post_init__(self):
        super().__post_init__()
        self.events.object_spawn.params['sweep_direction'] = 'left'
        self.terminations.arm_joint_velocity = None


@configclass
class WrenchRefactorV0_1EnvCfg(WrenchRefactorEnvCfg):
    """Direct v0 child with uncapped policy-input action-rate cost."""
    def __post_init__(self):
        super().__post_init__()
        from ..mdp.reward_functions.wrench_refactor import policy_action_rate_penalty
        self.rewards.robot_action_rate_penalty.func = policy_action_rate_penalty


@configclass
class WrenchRefactorV1_1EnvCfg(WrenchRefactorV1EnvCfg):
    """Direct v1 child with uncapped policy-input action-rate cost."""
    def __post_init__(self):
        super().__post_init__()
        from ..mdp.reward_functions.wrench_refactor import policy_action_rate_penalty
        self.rewards.robot_action_rate_penalty.func = policy_action_rate_penalty


@configclass
class WrenchRefactorV2_1EnvCfg(WrenchRefactorV2EnvCfg):
    """Direct v2 child with uncapped policy-input action-rate cost."""
    def __post_init__(self):
        super().__post_init__()
        from ..mdp.reward_functions.wrench_refactor import policy_action_rate_penalty
        self.rewards.robot_action_rate_penalty.func = policy_action_rate_penalty


@configclass
class WrenchRefactorV3_1EnvCfg(WrenchRefactorV3EnvCfg):
    """Direct v3 child with uncapped policy-input action-rate cost."""
    def __post_init__(self):
        super().__post_init__()
        from ..mdp.reward_functions.wrench_refactor import policy_action_rate_penalty
        self.rewards.robot_action_rate_penalty.func = policy_action_rate_penalty


@configclass
class WrenchRefactorRelativeEnvCfg(WrenchRefactorV3_1EnvCfg):
    """v3_1 physics/reset with shared, predominantly EEF-relative observations."""
    observations: WrenchRefactorRelativeObservationsCfg = WrenchRefactorRelativeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Vendor right-hand URDF: all twelve revolute joints effort=10, velocity=1.
        self.scene.robot.actuators['hand'].effort_limit_sim = 10.0
        self.rewards.target_goal_first_entry_bonus.weight = 10.0
