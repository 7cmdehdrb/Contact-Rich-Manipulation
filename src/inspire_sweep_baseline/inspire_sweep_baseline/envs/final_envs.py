"""Compose Final v0 independently, preserving Holding OSC ObjectRandom v3."""

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass

from ..assets.realistic_spawners import (
    RealisticRobotSpawnerCfg,
    replace_friction_spawner,
)
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.final import FinalSceneCfg
from ..mdp.actions.final import FinalActionsCfg, FinalOSCActionCfg
from ..mdp.commands.final import FinalCommandsCfg
from ..mdp.curricula.final import FinalCurriculumCfg
from ..mdp.events.final import FinalEventsCfg
from ..mdp.observations.final import FinalObservationsCfg
from ..mdp.rewards.final import FinalRewardsCfg
from ..mdp.terminations.final import FinalTerminationsCfg


@configclass
class FinalEnvCfg(ManagerBasedRLEnvCfg):
    scene: FinalSceneCfg = FinalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: FinalObservationsCfg = FinalObservationsCfg()
    actions: FinalActionsCfg = FinalActionsCfg()
    commands: FinalCommandsCfg = FinalCommandsCfg()
    rewards: FinalRewardsCfg = FinalRewardsCfg()
    terminations: FinalTerminationsCfg = FinalTerminationsCfg()
    events: FinalEventsCfg = FinalEventsCfg()
    curriculum: FinalCurriculumCfg = FinalCurriculumCfg()
    obstacle_observation_slots: int = 16
    viewer: ViewerCfg = ViewerCfg(eye=(3.5, 3.5, 3.5), lookat=(0.0, 0.0, 0.3))
    max_gray_obstacles: int = 8
    max_white_obstacles: int = 8

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.scene.robot.actuators["arm"].velocity_limit_sim = dict(
            zip(ARM_JOINT_NAMES, (2.0, 2.0, 2.0, 4.0, 4.0, 4.0))
        )
        defaults = self.sim.physics_material
        self.scene.robot.spawn = replace_friction_spawner(
            self.scene.robot.spawn,
            RealisticRobotSpawnerCfg,
            defaults,
        )
        self.scene.shelf.spawn.friction_defaults = defaults.copy()
        for obj in self.scene.object_collection.rigid_objects.values():
            obj.spawn.friction_defaults = defaults.copy()
        self.scene.configure_population_capacity(
            self.max_gray_obstacles,
            self.max_white_obstacles,
            self.obstacle_observation_slots,
        )
        self.events.object_spawn.params["gray_obstacles"] = min(
            1, self.max_gray_obstacles
        )
        self.events.object_spawn.params["white_obstacles"] = min(
            1, self.max_white_obstacles
        )
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action.body_name = eef.prim_path.rsplit("/", 1)[-1]
        self.actions.arm_action.body_offset = FinalOSCActionCfg.OffsetCfg(
            pos=eef.offset.pos,
            rot=eef.offset.rot,
        )
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0


from ..assets.scenes.final import FinalV1SceneCfg
from ..mdp.curricula.final import FinalV1CurriculumCfg
from ..mdp.events.final import FinalV1EventsCfg, FinalV2EventsCfg
from ..mdp.observations.final import FinalV1ObservationsCfg
from ..mdp.rewards.final import FinalV1RewardsCfg
from ..mdp.terminations.final import FinalV1TerminationsCfg


@configclass
class FinalV1EnvCfg(FinalEnvCfg):
    scene: FinalV1SceneCfg = FinalV1SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: FinalV1ObservationsCfg = FinalV1ObservationsCfg()
    rewards: FinalV1RewardsCfg = FinalV1RewardsCfg()
    terminations: FinalV1TerminationsCfg = FinalV1TerminationsCfg()
    events: FinalV1EventsCfg = FinalV1EventsCfg()
    curriculum: FinalV1CurriculumCfg = FinalV1CurriculumCfg()
    obstacle_observation_slots: int = 20
    max_gray_obstacles: int = 10
    max_white_obstacles: int = 10
    target_bounding_box: bool = False


@configclass
class FinalV2EnvCfg(FinalV1EnvCfg):
    events: FinalV2EventsCfg = FinalV2EventsCfg()


@configclass
class FinalV2TestEnvCfg(FinalV2EnvCfg):
    """Inspect v2 with its final requested population from the first reset."""

    def __post_init__(self):
        super().__post_init__()
        population = self.curriculum.obstacle_population.params
        self.events.object_spawn.params['gray_obstacles'] = population['final_gray']
        self.events.object_spawn.params['white_obstacles'] = population['final_white']
        self.curriculum.obstacle_population = None
