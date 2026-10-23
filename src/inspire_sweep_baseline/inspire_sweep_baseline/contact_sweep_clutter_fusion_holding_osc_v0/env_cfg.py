"""Holding v0 with relative OSC commands in the robot base frame."""

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from ..assets.robot import ARM_JOINT_NAMES
from ..contact_sweep_clutter_fusion_holding_v0.env_cfg import FusionHoldingV0EnvCfg


def configure_robot_base_osc(cfg):
    """Install the common Holding OSC controller without changing task managers."""
    eef = cfg.scene.ee_frame.target_frames[0]
    cfg.actions.arm_action = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=eef.prim_path.rsplit("/", 1)[-1],
        body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(
            pos=eef.offset.pos,
            rot=eef.offset.rot,
        ),
        # With no task frame, native pose_rel deltas use the robot root axes.
        task_frame_rel_path=None,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            motion_stiffness_task=100.0,
            motion_damping_ratio_task=1.0,
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
        ),
        position_scale=1.0,
        orientation_scale=1.0,
    )
    # OSC supplies joint efforts; implicit joint PD must not compete with it.
    cfg.scene.robot.actuators["arm"].stiffness = 0.0
    cfg.scene.robot.actuators["arm"].damping = 0.0


@configclass
class FusionHoldingOSCV0EnvCfg(FusionHoldingV0EnvCfg):
    """Replace only Arm action and the competing joint PD gains."""

    def __post_init__(self):
        super().__post_init__()
        configure_robot_base_osc(self)
