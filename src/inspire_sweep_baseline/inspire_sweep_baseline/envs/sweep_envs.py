"""Self-contained baseline configuration, derived from the v2-mass behavior.

Inherited baseline experiment varying only object mass and contact friction.

Realistic physics with an initial-position Actor and privileged live Critic.

Baseline mechanics with one independent binary observation per Hand pad."""

from ..assets.scenes.realistic_physics import apply_realistic_physics
from ..assets.scenes.sweep import SweepSceneCfg
from ..assets.scenes.sweep import TactileSweepSceneCfg
from ..mdp.actions.sweep import ActionsCfg
from ..mdp.commands.sweep import CommandsCfg
from ..mdp.events.sweep import EventsCfg
from ..mdp.events.sweep import RealisticPhysicsEventsCfg
from ..mdp.observation_functions.asymmetric_observations import InitialObjectPositionInRoot
from ..mdp.observations.sweep import InitialPrivilegedObservationsCfg
from ..mdp.observations.sweep import ObservationsCfg
from ..mdp.observations.sweep import TactileObservationsCfg
from ..mdp.rewards.sweep import RewardsCfg
from ..mdp.terminations.sweep import TerminationsCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg

@configclass
class InspireSweepEnvCfg(ManagerBasedRLEnvCfg):
    scene: SweepSceneCfg = SweepSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
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


@configclass
class InspireSweepRealisticPhysicsEnvCfg(InspireSweepEnvCfg):
    events: RealisticPhysicsEventsCfg = RealisticPhysicsEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_realistic_physics(self)


@configclass
class InspireSweepRealisticInitialPrivilegedEnvCfg(InspireSweepRealisticPhysicsEnvCfg):
    observations: InitialPrivilegedObservationsCfg = InitialPrivilegedObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Preserve the term's position in the inherited 41D policy vector.
        target = self.observations.policy.target_obs_state
        target.func = InitialObjectPositionInRoot
        target.params = {"command_name": "target_goal_pos"}
        # A single initial measurement has a single error, fixed until reset.
        target.noise = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=ConstantNoiseCfg(bias=0.0),
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),
            sample_bias_per_component=True,
        )


@configclass
class InspireSweepTactileEnvCfg(InspireSweepEnvCfg):
    scene: TactileSweepSceneCfg = TactileSweepSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TactileObservationsCfg = TactileObservationsCfg()
