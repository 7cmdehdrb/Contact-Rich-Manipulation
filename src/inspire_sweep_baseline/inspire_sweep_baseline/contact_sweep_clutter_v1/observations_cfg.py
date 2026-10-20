"""Initial target XYZ for the actor; current target XYZ only for the critic."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg

from ..contact_sweep_clutter.observations_cfg import ContactSweepClutterObservationsCfg
from ..mdp.asymmetric_observations import InitialObjectPositionInRoot
from ..mdp.observations import object_position_in_root
from .mdp import SharedTactileContactBits


@configclass
class ContactSweepClutterV1ObservationsCfg(ContactSweepClutterObservationsCfg):
    @configclass
    class PrivilegedCfg(ObservationGroupCfg):
        target_position = ObservationTermCfg(func=object_position_in_root)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged: PrivilegedCfg = PrivilegedCfg()

    def __post_init__(self):
        target = self.policy.target_obs_state
        target.func = InitialObjectPositionInRoot
        target.params = {"command_name": "target_goal_pos"}
        # Match Initial Privileged: one initial measurement and one fixed error.
        target.noise = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=ConstantNoiseCfg(bias=0.0),
            bias_noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="abs"),
            sample_bias_per_component=True,
        )
        self.policy.tactile.func = SharedTactileContactBits
