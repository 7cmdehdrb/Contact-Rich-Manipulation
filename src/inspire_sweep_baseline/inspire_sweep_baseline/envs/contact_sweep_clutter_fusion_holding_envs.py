"""Direct v1_0 configuration derivative with independent manager configs.

Extend Holding v0's initial-obstacle capacity without changing its task.

Holding v1 with synchronized goal tolerance and contact credit."""

from ..mdp.curricula.contact_sweep_clutter_fusion_holding import FusionHoldingV0CurriculumCfg
from ..mdp.curricula.contact_sweep_clutter_fusion_holding import FusionHoldingV2CurriculumCfg
from ..mdp.rewards.contact_sweep_clutter_fusion_holding import FusionHoldingV0RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion_holding import FusionHoldingV2RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion_holding import FusionHoldingV3RewardsCfg
from ..mdp.rewards.contact_sweep_clutter_fusion_holding import FusionHoldingV4RewardsCfg
from ..mdp.terminations.contact_sweep_clutter_fusion_holding import FusionHoldingV0TerminationsCfg
from ..mdp.terminations.contact_sweep_clutter_fusion_holding import FusionHoldingV2TerminationsCfg
from ..mdp.terminations.contact_sweep_clutter_fusion_holding import FusionHoldingV4TerminationsCfg
from .contact_sweep_clutter_fusion_envs import ContactSweepClutterFusionV1_0EnvCfg
from isaaclab.utils import configclass

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


@configclass
class FusionHoldingV1EnvCfg(FusionHoldingV0EnvCfg):
    # Keep the inherited 5 Gray + 5 White scene and population curriculum.
    # The extra slots support up to sixteen obstacles and are zero when unused.
    obstacle_observation_slots: int = 16


@configclass
class FusionHoldingV2EnvCfg(FusionHoldingV1EnvCfg):
    rewards: FusionHoldingV2RewardsCfg = FusionHoldingV2RewardsCfg()
    terminations: FusionHoldingV2TerminationsCfg = FusionHoldingV2TerminationsCfg()
    curriculum: FusionHoldingV2CurriculumCfg = FusionHoldingV2CurriculumCfg()


@configclass
class FusionHoldingV3EnvCfg(FusionHoldingV2EnvCfg):
    rewards: FusionHoldingV3RewardsCfg = FusionHoldingV3RewardsCfg()


@configclass
class FusionHoldingV4EnvCfg(FusionHoldingV3EnvCfg):
    rewards: FusionHoldingV4RewardsCfg = FusionHoldingV4RewardsCfg()
    terminations: FusionHoldingV4TerminationsCfg = FusionHoldingV4TerminationsCfg()
