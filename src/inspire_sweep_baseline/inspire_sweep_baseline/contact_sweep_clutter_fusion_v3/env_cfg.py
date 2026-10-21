"""Inherit Fusion v2, revise curricula, and keep push-fast fixed at 0.8 m/s."""

from isaaclab.utils import configclass

from ..contact_sweep_clutter_fusion_v2.env_cfg import ContactSweepClutterFusionV2EnvCfg
from .curriculum_cfg import ContactSweepClutterFusionV3CurriculumCfg


@configclass
class ContactSweepClutterFusionV3EnvCfg(ContactSweepClutterFusionV2EnvCfg):
    curriculum: ContactSweepClutterFusionV3CurriculumCfg = (
        ContactSweepClutterFusionV3CurriculumCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # 기존 0.8에서 1.5로 변경 -> 취소
        self.terminations.push_fast.params["speed_condition"] = 0.8
