"""Renewal variants changing only the tactile sensor views and observation."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from .dex_omni_renewal_env_cfg import InspireDexSweepOmniRenewalEnvCfg
from .mdp.tactile_observations import ProjectedTactileContactBits
from .omni_renewal_env_cfg import InspireSweepOmniRenewalEnvCfg
from .tactile_env_cfg import TactileObservationsCfg, TactileSweepSceneCfg


TACTILE_THRESHOLD_N = 0.01


@configclass
class TactileOmniRenewalSceneCfg(TactileSweepSceneCfg):
    """Original Hand pads and localized parent contacts, as in probe validation.

    The source robot, fixed pad attachments and collision geometry are retained.
    Only sensor views change; the standalone's optional exposed test fixture,
    fixed palm-up posture and zero gravity are not part of the RL scene.
    """

    def __post_init__(self):
        frame_debug_vis = self.ee_frame.debug_vis
        super().__post_init__()
        self.hand_tactile.force_threshold = TACTILE_THRESHOLD_N
        # Preserve Renewal's frame visualization as well as its transforms.
        self.ee_frame.debug_vis = frame_debug_vis


@configclass
class TactileOmniRenewalObservationsCfg(TactileObservationsCfg):
    @configclass
    class PolicyCfg(TactileObservationsCfg.PolicyCfg):
        tactile = ObsTerm(
            func=ProjectedTactileContactBits,
            params={"sensor_name": "hand_tactile", "threshold_n": TACTILE_THRESHOLD_N},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class InspireTactileSweepOmniRenewalEnvCfg(InspireSweepOmniRenewalEnvCfg):
    """Sweep Renewal with 17 binary tactile channels instead of one (57D)."""

    scene: TactileOmniRenewalSceneCfg = TactileOmniRenewalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TactileOmniRenewalObservationsCfg = TactileOmniRenewalObservationsCfg()


@configclass
class InspireDexTactileSweepOmniRenewalEnvCfg(InspireDexSweepOmniRenewalEnvCfg):
    """Dex Sweep Renewal with the same 17 tactile channels and original rewards."""

    scene: TactileOmniRenewalSceneCfg = TactileOmniRenewalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TactileOmniRenewalObservationsCfg = TactileOmniRenewalObservationsCfg()
