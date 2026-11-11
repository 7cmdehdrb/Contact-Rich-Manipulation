"""Simple finger OSC variants: bounded pose and sensor-origin wrench control."""
import torch
from isaaclab.utils.math import quat_apply_inverse
from .contact_sweep_clutter_fusion_holding_osc import UR10OperationalSpaceControllerAction
from .wrench_refactor import WrenchRefactorOSCAction
from ...assets.hand_kinematics import finger_action_bounds, clip_finger_actions


class BoundedFingerOSCAction(WrenchRefactorOSCAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lower, upper = finger_action_bounds(self.device)
        self._finger_lower, self._finger_upper = lower[:6], upper[:6]

    def process_actions(self, actions):
        super().process_actions(clip_finger_actions(actions, self._finger_lower, self._finger_upper))


class SensorFrameWrenchOSCAction(UR10OperationalSpaceControllerAction):
    """Pure wrench OSC at the AXIA80 origin; retain the parent's effort guards.

    Adapted from contact_manipulation_symmetric's SensorFrameWrenchAction.
    Native OSC adds a root-origin moment, while PhysX supplies a COM geometric
    Jacobian. Correct the moment reference before that native transformation.
    Clip raw commands to the reference SI-200-8 ranges before native scaling.
    ActionManager/PPO samples remain unchanged.
    """

    def __init__(self, cfg, env):
        if (cfg.controller_cfg.target_types != ['wrench_abs']
                or cfg.controller_cfg.impedance_mode != 'fixed'
                or cfg.body_offset is not None
                or cfg.task_frame_rel_path != f'Robot/{cfg.body_name}'):
            raise ValueError('Sensor wrench OSC requires fixed wrench_abs at the sensor body origin')
        super().__init__(cfg, env)
        if not torch.isfinite(self._wrench_scale).all() or (self._wrench_scale <= 0).any():
            raise ValueError('Wrench scale must be finite and positive')
        self._raw_wrench_limits = self._raw_actions.new_tensor((200., 200., 360., 8., 8., 8.)) / self._wrench_scale
        self._sensor_com_offset = self._asset.data.body_com_pos_b[:, self._ee_body_idx].clone()

    def _preprocess_actions(self, actions):
        super()._preprocess_actions(actions.clamp(-self._raw_wrench_limits, self._raw_wrench_limits))
        # Native set_command adds p_sensor/root x F_root. Subtract that and
        # the sensor-to-COM lever, leaving M_COM = R M_sensor - R c x R F.
        lever = (quat_apply_inverse(self._ee_pose_b[:, 3:7], self._ee_pose_b[:, :3])
                 + self._sensor_com_offset)
        self._processed_actions[:, 3:6] -= torch.cross(lever, self._processed_actions[:, :3], dim=-1)
