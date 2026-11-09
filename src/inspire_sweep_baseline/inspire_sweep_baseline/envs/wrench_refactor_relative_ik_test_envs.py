"""Independent IK-v0_1 scene/control and Holding OSC v2 reward environment."""
from ..mdp.rewards.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2_1TestRewardsCfg
from ..mdp.curricula.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestSceneCfg
from ..mdp.actions.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestActionsCfg
from ..mdp.observations.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestObservationsCfg
from ..mdp.commands.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestCommandsCfg
from ..mdp.rewards.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestRewardsCfg
from ..mdp.terminations.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestTerminationsCfg
from ..mdp.events.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestEventsCfg
from ..mdp.curricula.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestCurriculumCfg
from ..mdp.actions.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2TestOSCActionCfg

@configclass
class WrenchRefactorRelativeIKV0_2TestEnvCfg(ManagerBasedRLEnvCfg):
    scene: WrenchRefactorRelativeIKV0_2TestSceneCfg = WrenchRefactorRelativeIKV0_2TestSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: WrenchRefactorRelativeIKV0_2TestObservationsCfg = WrenchRefactorRelativeIKV0_2TestObservationsCfg()
    actions: WrenchRefactorRelativeIKV0_2TestActionsCfg = WrenchRefactorRelativeIKV0_2TestActionsCfg()
    commands: WrenchRefactorRelativeIKV0_2TestCommandsCfg = WrenchRefactorRelativeIKV0_2TestCommandsCfg()
    rewards: WrenchRefactorRelativeIKV0_2TestRewardsCfg = WrenchRefactorRelativeIKV0_2TestRewardsCfg()
    terminations: WrenchRefactorRelativeIKV0_2TestTerminationsCfg = WrenchRefactorRelativeIKV0_2TestTerminationsCfg()
    events: WrenchRefactorRelativeIKV0_2TestEventsCfg = WrenchRefactorRelativeIKV0_2TestEventsCfg()
    curriculum: WrenchRefactorRelativeIKV0_2TestCurriculumCfg = WrenchRefactorRelativeIKV0_2TestCurriculumCfg()
    obstacle_observation_slots: int = 16
    max_gray_obstacles: int = 3
    max_white_obstacles: int = 3
    target_bounding_box: bool = False
    goal_success_radius: float = 0.03
    goal_hold_outer_radius: float | None = None
    goal_reward_height_limit: float = 0.14
    debug_initial_ik_bounds: bool = True
    wrist_wrench_observation_limits: tuple = (150.0, 150.0, 150.0, 8.0, 8.0, 8.0)
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
        self.actions.arm_action.body_offset = WrenchRefactorRelativeIKV0_2TestOSCActionCfg.OffsetCfg(pos=eef.offset.pos, rot=eef.offset.rot)
        self.scene.robot.actuators['arm'].stiffness = 0.0
        self.scene.robot.actuators['arm'].damping = 0.0
        self.scene.robot.spawn.sensor_mass = 0.280
        # Right-hand source URDF: all 12 revolute joints effort=10 Nm, velocity=1 rad/s.
        self.scene.robot.actuators['hand'].effort_limit_sim = 10.0
        self.scene.robot.actuators['hand'].velocity_limit_sim = 1.0
        self.scene.robot.actuators['arm'].velocity_limit_sim = dict.fromkeys(ARM_JOINT_NAMES, 3.14)
        self.scene.robot.actuators['arm'].effort_limit_sim = dict(
            zip(ARM_JOINT_NAMES, (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)))




@configclass
class WrenchRefactorRelativeIKV0_2_1TestEnvCfg(WrenchRefactorRelativeIKV0_2TestEnvCfg):
    rewards: WrenchRefactorRelativeIKV0_2_1TestRewardsCfg = WrenchRefactorRelativeIKV0_2_1TestRewardsCfg()
    curriculum: WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg = WrenchRefactorRelativeIKV0_2_1TestCurriculumCfg()


from ..assets.scenes.wrench_refactor_relative_ik_test import GroupedResetObjectSpawnerCfg
from ..mdp.events.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2_1ResetTestEventsCfg


@configclass
class WrenchRefactorRelativeIKV0_2_1ResetTestEnvCfg(WrenchRefactorRelativeIKV0_2_1TestEnvCfg):
    events: WrenchRefactorRelativeIKV0_2_1ResetTestEventsCfg = WrenchRefactorRelativeIKV0_2_1ResetTestEventsCfg()
    reset_size_groups: int = 16
    reset_pool_capacity: int = 8
    reset_pool_refresh_interval: int = 32

    def __post_init__(self):
        super().__post_init__()
        for value in (self.reset_size_groups, self.reset_pool_capacity, self.reset_pool_refresh_interval):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Reset group/pool settings must be positive integers')
        for obj in self.scene.object_collection.rigid_objects.values():
            import copy
            values = copy.deepcopy(vars(obj.spawn))
            values.pop('func')
            obj.spawn = GroupedResetObjectSpawnerCfg(**values, reset_scale_groups=self.reset_size_groups)


from ..mdp.rewards.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2_2TestRewardsCfg


@configclass
class WrenchRefactorRelativeIKV0_2_2TestEnvCfg(WrenchRefactorRelativeIKV0_2_1TestEnvCfg):
    rewards: WrenchRefactorRelativeIKV0_2_2TestRewardsCfg = WrenchRefactorRelativeIKV0_2_2TestRewardsCfg()


from ..mdp.rewards.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2_2_1TestRewardsCfg


@configclass
class WrenchRefactorRelativeIKV0_2_2_1TestEnvCfg(WrenchRefactorRelativeIKV0_2_2TestEnvCfg):
    rewards: WrenchRefactorRelativeIKV0_2_2_1TestRewardsCfg = WrenchRefactorRelativeIKV0_2_2_1TestRewardsCfg()


from ..mdp.rewards.wrench_refactor_relative_ik_test import WrenchRefactorRelativeIKV0_2_2_2TestRewardsCfg


@configclass
class WrenchRefactorRelativeIKV0_2_2_2TestEnvCfg(WrenchRefactorRelativeIKV0_2_2_1TestEnvCfg):
    rewards: WrenchRefactorRelativeIKV0_2_2_2TestRewardsCfg = WrenchRefactorRelativeIKV0_2_2_2TestRewardsCfg()
