"""Direct v1_0 configuration derivative with independent manager configs."""
from isaaclab.utils import configclass
from ..contact_sweep_clutter_fusion_v1_0.env_cfg import ContactSweepClutterFusionV1_0EnvCfg
from .rewards_cfg import FusionHoldingV0RewardsCfg
from .terminations_cfg import FusionHoldingV0TerminationsCfg
from .curriculum_cfg import FusionHoldingV0CurriculumCfg


@configclass
class FusionHoldingV0EnvCfg(ContactSweepClutterFusionV1_0EnvCfg):
    max_gray_obstacles: int = 5
    max_white_obstacles: int = 5
    rewards: FusionHoldingV0RewardsCfg = FusionHoldingV0RewardsCfg()
    terminations: FusionHoldingV0TerminationsCfg = FusionHoldingV0TerminationsCfg()
    curriculum: FusionHoldingV0CurriculumCfg = FusionHoldingV0CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # The inherited v3 post-init sets 0.8; apply Holding's threshold last.
        self.terminations.push_fast.params['speed_condition'] = 1.0
