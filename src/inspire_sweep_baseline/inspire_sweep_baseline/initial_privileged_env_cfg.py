"""Realistic physics with an initial-position Actor and privileged live Critic."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from .env_cfg import ObservationsCfg
from .mdp.asymmetric_observations import InitialObjectPositionInRoot, live_object_state_in_root
from .realistic_physics_env_cfg import InspireSweepRealisticPhysicsEnvCfg


@configclass
class InitialPrivilegedObservationsCfg(ObservationsCfg):
    @configclass
    class PrivilegedCfg(ObsGroup):
        target_state = ObsTerm(func=live_object_state_in_root)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()


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
