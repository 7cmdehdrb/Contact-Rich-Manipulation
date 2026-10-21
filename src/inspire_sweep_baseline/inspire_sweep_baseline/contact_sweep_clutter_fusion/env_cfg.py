"""Independent Isaac Lab environment and manager configurations for fusion."""

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.utils import configclass

from ..assets.realistic_spawners import RealisticRobotSpawnerCfg, replace_friction_spawner
from ..assets.robot import ARM_JOINT_NAMES
from .actions_cfg import ContactSweepClutterFusionActionsCfg
from .commands_cfg import ContactSweepClutterFusionCommandsCfg
from .curriculum_cfg import ContactSweepClutterFusionCurriculumCfg
from .events_cfg import ContactSweepClutterFusionEventsCfg
from .observations_cfg import ContactSweepClutterFusionObservationsCfg
from .rewards_cfg import ContactSweepClutterFusionRewardsCfg
from .scene_cfg import ContactSweepClutterFusionSceneCfg
from .terminations_cfg import ContactSweepClutterFusionTerminationsCfg


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
