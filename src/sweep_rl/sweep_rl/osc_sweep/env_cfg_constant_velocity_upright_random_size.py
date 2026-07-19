"""ConstantVelocity-v0 with gripper-interior exclusion.

The registered environment name is retained for checkpoint and script
compatibility.  It inherits ``UR5eOscSweepConstantVelocityEnvCfg``, replaces
only the pre-contact pose target, and adds one failure termination.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import EEF_ENTITY_CFG
from .env_cfg_constant_velocity import (
    ConstantVelocityRewardsCfg,
    ConstantVelocityTerminationsCfg,
    UR5eOscSweepConstantVelocityEnvCfg,
)


@configclass
class GripperExclusionRewardsCfg(ConstantVelocityRewardsCfg):
    """Base rewards with a pre-contact target outside the gripper gap."""

    push_pose_error = RewTerm(
        func=mdp.external_pad_precontact_pose_error,
        weight=-1.0,
        params={
            "command_name": "desired_motion",
            "distance_scale": 0.10,
            "stand_off": 0.065,
            "pad_center_offset": 0.055,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class GripperExclusionTerminationsCfg(ConstantVelocityTerminationsCfg):
    """Base terminations plus failure on inserting the cube between fingers."""

    object_inside_gripper = DoneTerm(
        func=mdp.object_inside_gripper,
        time_out=False,
        params={
            # EEF-local XYZ exclusion volume.  Local Y spans the open gap
            # between pad inner faces (+/-0.040 m). X/Z include the 0.030 m
            # cube half-size so entry is detected when the cube overlaps the
            # finger-depth region, not only after its center reaches the EEF.
            "center_half_extents": (0.040, 0.040, 0.058),
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg(
    UR5eOscSweepConstantVelocityEnvCfg
):
    """ConstantVelocity-v0 with external-pad approach and insertion failure."""

    rewards: GripperExclusionRewardsCfg = GripperExclusionRewardsCfg()
    terminations: GripperExclusionTerminationsCfg = (
        GripperExclusionTerminationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        # Treat insertion like every other safety failure. Without this, an
        # agent could deliberately terminate early to escape running costs.
        failure_names = self.rewards.failure_termination.params["term_names"]
        if "object_inside_gripper" not in failure_names:
            self.rewards.failure_termination.params["term_names"] = (
                *failure_names,
                "object_inside_gripper",
            )
