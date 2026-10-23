"""Contact Sweep is a direct Isaac Lab configuration, independent of Renewal Cfgs."""

from ..assets.realistic_spawners import RealisticRobotSpawnerCfg
from ..assets.realistic_spawners import replace_friction_spawner
from ..assets.robot import ARM_JOINT_NAMES
from ..assets.scenes.contact_sweep import ContactSweepSceneCfg
from ..mdp.actions.contact_sweep import ContactSweepActionsCfg
from ..mdp.commands.contact_sweep import ContactSweepCommandsCfg
from ..mdp.events.contact_sweep import ContactSweepEventsCfg
from ..mdp.observations.contact_sweep import ContactSweepObservationsCfg
from ..mdp.rewards.contact_sweep import ContactSweepRewardsCfg
from ..mdp.terminations.contact_sweep import ContactSweepTerminationsCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

@configclass
class ContactSweepEnvCfg(ManagerBasedRLEnvCfg):
    scene: ContactSweepSceneCfg = ContactSweepSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ContactSweepObservationsCfg = ContactSweepObservationsCfg()
    actions: ContactSweepActionsCfg = ContactSweepActionsCfg()
    commands: ContactSweepCommandsCfg = ContactSweepCommandsCfg()
    rewards: ContactSweepRewardsCfg = ContactSweepRewardsCfg()
    terminations: ContactSweepTerminationsCfg = ContactSweepTerminationsCfg()
    events: ContactSweepEventsCfg = ContactSweepEventsCfg()
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
        # Source-asset material binding; runtime reset only updates Tensor API buffers.
        defaults = self.sim.physics_material
        self.scene.robot.spawn = replace_friction_spawner(
            self.scene.robot.spawn, RealisticRobotSpawnerCfg, defaults,
        )
        self.scene.shelf.spawn.friction_defaults = defaults.copy()
        self.scene.object_collection.rigid_objects["mass_cylinder"].spawn.friction_defaults = defaults.copy()
