"""Six joint-position actions and two absolute Hand openness commands."""

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.utils import configclass

from ..assets.robot import ARM_JOINT_NAMES
from ..mdp.actions import HandOpennessActionCfg


@configclass
class ContactSweepClutterFusionActionsCfg:
    arm_action = JointPositionActionCfg(
        asset_name="robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True,
        scale=0.5, use_default_offset=True,
    )
    gripper_action = HandOpennessActionCfg(asset_name="robot")
