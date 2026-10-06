"""Contact-aware, fixed-gain Cartesian OSC for the outward V1 hand poses."""

from __future__ import annotations

import torch

from isaaclab.controllers import OperationalSpaceController
from isaaclab.utils.math import compute_pose_error

from ..action_math import quaternion_rotate
from ..assets.robot import PALM_SENSOR_BODY_NAMES
from ..impedance_math import cube_reaction_wrench_b, implicit_cartesian_impedance_efforts


class ContactCartesianOscController(OperationalSpaceController):
    """Predict measured palmar reactions while preserving input stiffness 200.

    The parent owns command transforms. The ActionTerm still owns target
    accumulation, measured C kinematics, gravity, and final effort clipping.
    No filtered force is used to synthesize an observation or reward.
    """

    def __init__(self, previous_controller, action):
        super().__init__(previous_controller.cfg, action.num_envs, action.device)
        self._action = action
        self._physics_dt = action._env.cfg.sim.dt
        self._stiffness = torch.tensor(action.cfg.motion_stiffness, device=action.device)
        self._damping_ratio = torch.tensor(action.cfg.motion_damping_ratio, device=action.device)
        ids, names = action._asset.find_bodies(list(PALM_SENSOR_BODY_NAMES), preserve_order=True)
        if tuple(names) != PALM_SENSOR_BODY_NAMES:
            raise ValueError("Contact OSC requires canonical palmar sensor body order")
        self._pad_ids = torch.tensor(ids, dtype=torch.long, device=action.device)
        self._external_wrench = torch.zeros((action.num_envs, 6), device=action.device)

    def reset_contact_history(self, env_ids=None):
        self._external_wrench[slice(None) if env_ids is None else env_ids] = 0.

    def _contact_reaction(self):
        action, env = self._action, self._action._env
        robot = action._asset
        forces = env.scene["cube_palm_contacts"].data.force_matrix_w[:, 0]
        safe = env.event_manager.get_term_cfg("safe_hand").func
        centers_b = safe.collision_bounds.centers[self._pad_ids]
        centers_w = robot.data.body_link_pos_w[:, self._pad_ids] + quaternion_rotate(
            robot.data.body_link_quat_w[:, self._pad_ids], centers_b.expand(action.num_envs, -1, -1)
        )
        root_quaternion = robot.data.root_link_quat_w
        control_w = robot.data.root_link_pos_w + quaternion_rotate(root_quaternion, action._c_pose_b[:, :3])
        measured = cube_reaction_wrench_b(forces, centers_w, control_w, root_quaternion)
        for value, limit in ((measured[:, :3], 20.), (measured[:, 3:], 2.)):
            value *= (limit / value.norm(dim=-1, keepdim=True).clamp_min(limit)).clamp_max(1.)
        self._external_wrench = .5 * self._external_wrench + .5 * measured
        return self._external_wrench

    def compute(self, jacobian_b, current_ee_pose_b=None, current_ee_vel_b=None,
                mass_matrix=None, gravity=None, **kwargs):
        if mass_matrix is None or current_ee_pose_b is None or current_ee_vel_b is None:
            raise ValueError("Contact-aware Cartesian OSC requires mass, C pose and C twist")
        errors = compute_pose_error(
            current_ee_pose_b[:, :3], current_ee_pose_b[:, 3:],
            self.desired_ee_pose_b[:, :3], self.desired_ee_pose_b[:, 3:], rot_error_type="axis_angle",
        )
        return implicit_cartesian_impedance_efforts(
            jacobian_b, mass_matrix, torch.cat(errors, -1), current_ee_vel_b,
            self._stiffness, self._damping_ratio, self._physics_dt,
            gravity=gravity, external_wrench=self._contact_reaction(),
        )


__all__ = ["ContactCartesianOscController"]
