"""Independent Isaac Lab environment and manager configurations for fusion.

Inherit Fusion v0 and remove only its EEF orientation reward.

Inherit Fusion v2, revise curricula, and keep push-fast fixed at 0.8 m/s.

Extend Fusion v3 with explicit, independent manager configurations.

Parameter variant with explicit reward and curriculum configurations.

Acceleration experiment derived from v1_1, with independent managers.

Force curriculum experiment derived from v1_1, with independent managers.

Goal and contact-width experiment derived from v1_3, with independent managers.

Direct v1.0 derivative; no dependency on v1.0 variants or v1.1–v1.3."""

from ..assets.realistic_spawners import RealisticRobotSpawnerCfg
from ..assets.realistic_spawners import replace_friction_spawner
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.contact_sweep_clutter_fusion import ContactSweepClutterFusionSceneCfg
from ..assets.scenes.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0SceneCfg
from ..mdp.actions.contact_sweep_clutter_fusion import ContactSweepClutterFusionActionsCfg
from ..mdp.commands.contact_sweep_clutter_fusion import ContactSweepClutterFusionCommandsCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionCurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_1CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_1_1CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_1_2CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_2CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_3CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_3_1CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_4CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion import ContactSweepClutterFusionV3CurriculumCfg
from ..mdp.events.contact_sweep_clutter_fusion import ContactSweepClutterFusionEventsCfg
from ..mdp.events.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0EventsCfg
from ..mdp.observations.contact_sweep_clutter_fusion import ContactSweepClutterFusionObservationsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionRewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_1RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_1_1RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_1_2RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_2RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_3RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_3_1RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_4RewardsCfg
from ..mdp.terminations.contact_sweep_clutter_fusion import ContactSweepClutterFusionTerminationsCfg
from ..mdp.terminations.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_0TerminationsCfg
from ..mdp.terminations.contact_sweep_clutter_fusion import ContactSweepClutterFusionV1_4TerminationsCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass
import math as math

@configclass
class ContactSweepClutterFusionEnvCfg(ManagerBasedRLEnvCfg):
    scene: ContactSweepClutterFusionSceneCfg = ContactSweepClutterFusionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ContactSweepClutterFusionObservationsCfg = ContactSweepClutterFusionObservationsCfg()
    actions: ContactSweepClutterFusionActionsCfg = ContactSweepClutterFusionActionsCfg()
    commands: ContactSweepClutterFusionCommandsCfg = ContactSweepClutterFusionCommandsCfg()
    rewards: ContactSweepClutterFusionRewardsCfg = ContactSweepClutterFusionRewardsCfg()
    terminations: ContactSweepClutterFusionTerminationsCfg = ContactSweepClutterFusionTerminationsCfg()
    events: ContactSweepClutterFusionEventsCfg = ContactSweepClutterFusionEventsCfg()
    curriculum: ContactSweepClutterFusionCurriculumCfg = ContactSweepClutterFusionCurriculumCfg()
    obstacle_observation_slots: int = 10
    viewer: ViewerCfg = ViewerCfg(eye=(3.5, 3.5, 3.5), lookat=(0.0, 0.0, 0.3))

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
            self.scene.robot.spawn, RealisticRobotSpawnerCfg, defaults,
        )
        self.scene.shelf.spawn.friction_defaults = defaults.copy()
        for obj in self.scene.object_collection.rigid_objects.values():
            obj.spawn.friction_defaults = defaults.copy()


@configclass
class ContactSweepClutterFusionV2EnvCfg(ContactSweepClutterFusionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.eef_orientation = None


@configclass
class ContactSweepClutterFusionV3EnvCfg(ContactSweepClutterFusionV2EnvCfg):
    curriculum: ContactSweepClutterFusionV3CurriculumCfg = (
        ContactSweepClutterFusionV3CurriculumCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # 기존 0.8에서 1.5로 변경 -> 취소
        self.terminations.push_fast.params["speed_condition"] = 0.8


@configclass
class ContactSweepClutterFusionV1_0EnvCfg(ContactSweepClutterFusionV3EnvCfg):
    max_gray_obstacles: int = 3
    max_white_obstacles: int = 3
    scene: ContactSweepClutterFusionV1_0SceneCfg = ContactSweepClutterFusionV1_0SceneCfg(
        num_envs=4096, env_spacing=2.5,
    )
    rewards: ContactSweepClutterFusionV1_0RewardsCfg = ContactSweepClutterFusionV1_0RewardsCfg()
    terminations: ContactSweepClutterFusionV1_0TerminationsCfg = ContactSweepClutterFusionV1_0TerminationsCfg()
    curriculum: ContactSweepClutterFusionV1_0CurriculumCfg = ContactSweepClutterFusionV1_0CurriculumCfg()
    events: ContactSweepClutterFusionV1_0EventsCfg = ContactSweepClutterFusionV1_0EventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.configure_population_capacity(
            self.max_gray_obstacles, self.max_white_obstacles, self.obstacle_observation_slots,
        )
        self.events.object_spawn.params["gray_obstacles"] = min(1, self.max_gray_obstacles)
        self.events.object_spawn.params["white_obstacles"] = min(1, self.max_white_obstacles)
        # Also correct before the first runner update/reset.
        self.events.object_spawn.params["excluded_x_half_angle"] = math.radians(30.0)


@configclass
class ContactSweepClutterFusionV1_1EnvCfg(ContactSweepClutterFusionV1_0EnvCfg):
    rewards: ContactSweepClutterFusionV1_1RewardsCfg = ContactSweepClutterFusionV1_1RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_1CurriculumCfg = ContactSweepClutterFusionV1_1CurriculumCfg()


@configclass
class ContactSweepClutterFusionV1_1_1EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_1_1RewardsCfg = ContactSweepClutterFusionV1_1_1RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_1_1CurriculumCfg = ContactSweepClutterFusionV1_1_1CurriculumCfg()


@configclass
class ContactSweepClutterFusionV1_1_2EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_1_2RewardsCfg = ContactSweepClutterFusionV1_1_2RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_1_2CurriculumCfg = ContactSweepClutterFusionV1_1_2CurriculumCfg()


@configclass
class ContactSweepClutterFusionV1_2EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_2RewardsCfg = ContactSweepClutterFusionV1_2RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_2CurriculumCfg = ContactSweepClutterFusionV1_2CurriculumCfg()


@configclass
class ContactSweepClutterFusionV1_3EnvCfg(ContactSweepClutterFusionV1_1EnvCfg):
    rewards: ContactSweepClutterFusionV1_3RewardsCfg = ContactSweepClutterFusionV1_3RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_3CurriculumCfg = ContactSweepClutterFusionV1_3CurriculumCfg()


@configclass
class ContactSweepClutterFusionV1_3_1EnvCfg(ContactSweepClutterFusionV1_3EnvCfg):
    rewards: ContactSweepClutterFusionV1_3_1RewardsCfg = ContactSweepClutterFusionV1_3_1RewardsCfg()
    curriculum: ContactSweepClutterFusionV1_3_1CurriculumCfg = ContactSweepClutterFusionV1_3_1CurriculumCfg()


@configclass
class ContactSweepClutterFusionV1_4EnvCfg(ContactSweepClutterFusionV1_0EnvCfg):
    max_gray_obstacles: int = 5
    max_white_obstacles: int = 5
    rewards: ContactSweepClutterFusionV1_4RewardsCfg = ContactSweepClutterFusionV1_4RewardsCfg()
    terminations: ContactSweepClutterFusionV1_4TerminationsCfg = ContactSweepClutterFusionV1_4TerminationsCfg()
    curriculum: ContactSweepClutterFusionV1_4CurriculumCfg = ContactSweepClutterFusionV1_4CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # v3's inherited post-init explicitly restores 0.8; override it last.
        self.terminations.push_fast.params["speed_condition"] = 1.0
