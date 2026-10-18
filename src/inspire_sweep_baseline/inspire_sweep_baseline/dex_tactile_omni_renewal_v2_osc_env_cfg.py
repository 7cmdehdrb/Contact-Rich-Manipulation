"""Dex Tactile v2 with bounded relative OSC; retain all task mechanics."""

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.utils import configclass

from .assets.robot import ARM_JOINT_NAMES
from .dex_tactile_omni_renewal_v2_env_cfg import InspireDexTactileSweepOmniRenewalV2EnvCfg
from .mdp.osc_actions import BoundedOperationalSpaceControllerActionCfg


@configclass
class InspireDexTactileSweepOmniRenewalV2OSCEnvCfg(InspireDexTactileSweepOmniRenewalV2EnvCfg):
    """Replace only Arm control, with 100 stiffness and small pose deltas."""

    def __post_init__(self):
        super().__post_init__()
        eef = self.scene.ee_frame.target_frames[0]
        self.actions.arm_action = BoundedOperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=list(ARM_JOINT_NAMES),
            body_name=eef.prim_path.rsplit("/", 1)[-1],
            body_offset=BoundedOperationalSpaceControllerActionCfg.OffsetCfg(
                pos=eef.offset.pos,
                rot=eef.offset.rot,
            ),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                inertial_dynamics_decoupling=True,
                gravity_compensation=True,
            ),
            position_scale=0.005,
            orientation_scale=0.025,
            position_command_limit=0.005,
            orientation_command_limit=0.025,
        )
        # As in Renewal OSC, joint PD must not compete with OSC efforts.
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0
