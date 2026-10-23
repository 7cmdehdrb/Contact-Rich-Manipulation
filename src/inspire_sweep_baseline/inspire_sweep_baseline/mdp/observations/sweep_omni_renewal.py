"""Renewal variants with 17 tactile observations and shared exponential costs."""

from ..common.sweep_omni_renewal import TACTILE_THRESHOLD_N
from ..observation_functions.tactile_observations import ProjectedTactileContactBits
from .sweep import TactileObservationsCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

@configclass
class TactileOmniRenewalObservationsCfg(TactileObservationsCfg):
    @configclass
    class PolicyCfg(TactileObservationsCfg.PolicyCfg):
        tactile = ObsTerm(
            func=ProjectedTactileContactBits,
            params={"sensor_name": "hand_tactile", "threshold_n": TACTILE_THRESHOLD_N},
        )

    policy: PolicyCfg = PolicyCfg()
