"""Keep the 57D policy group and expose 40D initial geometry to the model encoder."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from ..contact_sweep.observations_cfg import ContactSweepObservationsCfg
from .mdp import initial_obstacle_positions


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
