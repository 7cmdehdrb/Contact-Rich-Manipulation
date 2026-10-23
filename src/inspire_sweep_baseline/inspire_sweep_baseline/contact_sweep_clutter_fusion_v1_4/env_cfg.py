"""Direct v1.0 derivative; no dependency on v1.0 variants or v1.1–v1.3."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v1_0.env_cfg import ContactSweepClutterFusionV1_0EnvCfg
from .curriculum_cfg import ContactSweepClutterFusionV1_4CurriculumCfg
from .rewards_cfg import ContactSweepClutterFusionV1_4RewardsCfg
from .terminations_cfg import ContactSweepClutterFusionV1_4TerminationsCfg


@configclass
class ContactSweepClutterFusionV1_4EnvCfg(ContactSweepClutterFusionV1_0EnvCfg):
    max_gray_obstacles: int = 5
    max_white_obstacles: int = 5
    rewards: ContactSweepClutterFusionV1_4RewardsCfg = ContactSweepClutterFusionV1_4RewardsCfg()
    terminations: ContactSweepClutterFusionV1_4TerminationsCfg = ContactSweepClutterFusionV1_4TerminationsCfg()
    curriculum: ContactSweepClutterFusionV1_4CurriculumCfg = ContactSweepClutterFusionV1_4CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # v3's inherited post-init explicitly restores 0.8; override it last.
        self.terminations.push_fast.params["speed_condition"] = 1.0
