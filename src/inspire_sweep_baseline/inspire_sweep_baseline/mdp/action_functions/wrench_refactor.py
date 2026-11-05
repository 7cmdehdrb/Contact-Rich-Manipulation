"""Bounded OSC with the offset EEF as both control point and command frame."""

from .contact_sweep_clutter_fusion_holding_osc import UR10OperationalSpaceControllerAction
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, skew_symmetric_matrix
import torch


class WrenchRefactorOSCAction(UR10OperationalSpaceControllerAction):
    def __init__(self, cfg, env):
        if cfg.task_frame_rel_path is not None or cfg.controller_cfg.target_types != ["pose_rel"]:
            raise ValueError("Refactor OSC requires pose_rel and the EEF task frame")
        super().__init__(cfg, env)

    def _compute_task_frame_pose(self):
        # Isaac Lab updates _ee_pose_b including body_offset before this call.
        # Freeze the command frame over the decimation substeps; do not alias
        # the live EE pose, which is refreshed on each physics step.
        self._task_frame_pose_b = self._ee_pose_b.clone()

    def _compute_ee_jacobian(self):
        # Both Jacobian blocks use ROOT axes. The body-local lever arm must be
        # rotated to root axes before shifting the linear block to the EEF.
        self._jacobian_b[:] = self.jacobian_b
        if self.cfg.body_offset is not None:
            data = self._asset.data
            body_in_root = quat_mul(quat_conjugate(data.root_quat_w),
                                    data.body_link_quat_w[:, self._ee_body_idx])
            lever = quat_apply(body_in_root, self._offset_pos)
            self._jacobian_b[:, :3] -= torch.bmm(skew_symmetric_matrix(lever), self._jacobian_b[:, 3:])
            # Changing the controlled frame's origin/orientation does not
            # change angular velocity expressed in root axes.

    def _compute_ee_velocity(self):
        # body_vel_w is a COM-velocity alias in this Isaac Lab version. Use
        # link-origin velocity before applying the link-to-EEF lever arm.
        data = self._asset.data
        link_q = data.body_link_quat_w[:, self._ee_body_idx]
        velocity = data.body_link_vel_w[:, self._ee_body_idx].clone()
        if self.cfg.body_offset is not None:
            lever_w = quat_apply(link_q, self._offset_pos)
            velocity[:, :3] += torch.cross(velocity[:, 3:], lever_w, dim=-1)
        self._ee_vel_w[:] = velocity
        relative = velocity - data.root_link_vel_w
        self._ee_vel_b[:, :3] = quat_apply_inverse(data.root_quat_w, relative[:, :3])
        self._ee_vel_b[:, 3:] = quat_apply_inverse(data.root_quat_w, relative[:, 3:])
