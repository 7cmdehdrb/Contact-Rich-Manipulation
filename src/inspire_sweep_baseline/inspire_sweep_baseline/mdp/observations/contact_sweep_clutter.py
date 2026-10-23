"""Keep the 57D policy group and expose 40D initial geometry to the model encoder.

Initial target XYZ for the actor; current target XYZ only for the critic.

Share the same compensated physical sample between observations and reward."""

from ..observation_functions.asymmetric_observations import InitialObjectPositionInRoot
from ..observation_functions.contact_sweep_clutter import PureWristWrench
from ..observation_functions.contact_sweep_clutter import SharedTactileContactBits
from ..observation_functions.contact_sweep_clutter import initial_obstacle_positions
from ..observation_functions.sweep import object_position_in_root
from .contact_sweep import ContactSweepObservationsCfg
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import ConstantNoiseCfg
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg

@configclass
class ContactSweepClutterObservationsCfg(ContactSweepObservationsCfg):
    @configclass
    class ObstaclesCfg(ObservationGroupCfg):
        initial_positions = ObservationTermCfg(func=initial_obstacle_positions)

        def __post_init__(self):
            # Noise is sampled by the reset event once, not on each observation.
            self.enable_corruption = False
            self.concatenate_terms = True

    obstacles: ObstaclesCfg = ObstaclesCfg()


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


@configclass
class ContactSweepClutterV2ObservationsCfg(ContactSweepClutterV1ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.wrist_wrench.func = PureWristWrench
        self.policy.wrist_wrench.params = {}
