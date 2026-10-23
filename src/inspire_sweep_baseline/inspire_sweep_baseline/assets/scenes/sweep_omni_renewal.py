"""Renewal variants with 17 tactile observations and shared exponential costs."""

from ...mdp.common.sweep_omni_renewal import TACTILE_THRESHOLD_N
from .sweep import TactileSweepSceneCfg
from isaaclab.utils import configclass

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
