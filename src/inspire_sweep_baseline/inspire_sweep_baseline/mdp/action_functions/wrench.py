"""Six-axis sensor-frame wrench OSC, adapted from Isaac Lab's force-only P loop.

Reference: IsaacLab/source/isaaclab/test/controllers/test_operational_space.py
closed-loop wrench test and isaaclab/controllers/operational_space.py FF + P equation.
The upstream ContactSensor path cannot feed back moments. This path uses the
existing Axia measurement joint, without changing the simulator or assets.
"""

from __future__ import annotations

import math
import torch
from isaaclab.managers import ActionTerm
from isaaclab.utils.math import matrix_from_quat

from ..common.compensated_wrench import PureWrenchReader
from ..common.wrench_control import sensor_wrench_efforts


class SensorFrameWrenchAction(ActionTerm):
    """Six sensor-axis wrench targets at policy rate, F/T feedback at physics rate.

    Uses the axia80_link origin, no EEF offset, no contact-force proxy, no pose
    target. Sensor axes follow the sensor at every substep. Observation/reward
    EEF frames and their compensated wrench samples remain untouched.
    """

    def __init__(self, cfg, env):
        c = cfg.controller_cfg
        if (list(c.target_types) != ['wrench_abs'] or c.impedance_mode != 'fixed'
                or tuple(c.motion_control_axes_task) != (0,) * 6
                or tuple(c.contact_wrench_control_axes_task) != (1,) * 6
                or c.nullspace_control != 'none' or c.inertial_dynamics_decoupling):
            raise ValueError('Sensor wrench OSC requires fixed, six-axis wrench-only control without motion/nullspace')
        if cfg.body_name != 'axia80_link' or cfg.body_offset is not None or cfg.task_frame_rel_path is not None:
            raise ValueError('Wrench OSC reference must be the axia80_link sensor frame without an offset')
        if c.contact_wrench_stiffness_task is None:
            raise ValueError('Closed-loop wrench control requires proportional gains')
        if (not math.isfinite(cfg.normalized_wrench_limit) or cfg.normalized_wrench_limit <= 0
                or not math.isfinite(cfg.effort_limit_scale) or cfg.effort_limit_scale <= 0):
            raise ValueError('Wrench and effort limits must be positive and finite')
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        bodies, _ = self._asset.find_bodies(cfg.body_name)
        if len(bodies) != 1 or len(self._joint_ids) != 6 or not self._asset.is_fixed_base:
            raise ValueError('Wrench-v1 requires one sensor body and a fixed-base six-joint arm')
        self._body_idx = bodies[0]
        self._jacobian_body_idx = self._body_idx - 1
        if self._jacobian_body_idx < 0:
            raise ValueError('The sensor body cannot be the articulation root')
        self._p_gains = torch.as_tensor(c.contact_wrench_stiffness_task, device=self.device).float().expand(6).clone()
        self._wrench_scale = torch.as_tensor(cfg.wrench_scale, device=self.device).float().expand(6).clone()
        if (not torch.isfinite(self._p_gains).all() or (self._p_gains <= 0).any()
                or not torch.isfinite(self._wrench_scale).all() or (self._wrench_scale <= 0).any()):
            raise ValueError('All six feedback gains and wrench scales must be positive and finite')
        self._commanded_wrench_limit = torch.as_tensor(
            cfg.commanded_wrench_limit, device=self.device).float().expand(6).clone()
        if (not torch.isfinite(self._commanded_wrench_limit).all()
                or (self._commanded_wrench_limit <= 0).any()):
            raise ValueError('All six commanded wrench limits must be positive and finite')
        self._raw_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._current_wrench_s = torch.zeros_like(self._raw_actions)
        self._joint_efforts = torch.zeros_like(self._raw_actions)
        self._torque_saturated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._feedback_ready_after = torch.full((self.num_envs,), env._sim_step_counter + 1,
                                                device=self.device, dtype=torch.long)
        # Own physics-rate sample; never reuse the noisy policy observation or its control-rate cache.
        self._wrench_reader = PureWrenchReader(self._asset, gravity_w=env.cfg.sim.gravity)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    @property
    def current_wrench_s(self):
        return self._current_wrench_s.clone()

    @property
    def joint_efforts(self):
        return self._joint_efforts

    @property
    def torque_saturated(self):
        return self._torque_saturated

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(
            -self.cfg.normalized_wrench_limit, self.cfg.normalized_wrench_limit) * self._wrench_scale

    def apply_actions(self):
        data = self._asset.data
        position = data.body_link_pos_w[:, self._body_idx]
        orientation = data.body_link_quat_w[:, self._body_idx]
        # At apply_action k the last completed physics update is k-1. Skip the
        # first sample after each reset, without masking live rows in a partial reset.
        valid = self._env._sim_step_counter > self._feedback_ready_after
        self._current_wrench_s[:] = self._wrench_reader.read(
            position, orientation, valid_env_mask=valid).pure_eef
        # Despite the reader's historical field name, the supplied reference is
        # the SENSOR. It transports joint-plane -> sensor-center (12.7 mm), not EEF.
        jacobian_w = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobian_body_idx, :, :][:, :, self._joint_ids]
        sensor_from_world = matrix_from_quat(orientation).transpose(-1, -2)
        jacobian_s = torch.cat((sensor_from_world @ jacobian_w[:, :3],
                                sensor_from_world @ jacobian_w[:, 3:]), dim=1)
        gravity = None
        if self.cfg.controller_cfg.gravity_compensation:
            gravity = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        self._joint_efforts[:] = sensor_wrench_efforts(
            self._processed_actions, self._current_wrench_s, jacobian_s, self._p_gains,
            gravity=gravity, feedback_valid=valid,
            commanded_wrench_limit=self._commanded_wrench_limit)
        finite = torch.isfinite(self._joint_efforts).all(dim=1) & torch.isfinite(self._raw_actions).all(dim=1)
        self._joint_efforts[:] = torch.where(finite[:, None], self._joint_efforts, 0.0)
        limits = (data.joint_effort_limits[:, self._joint_ids] * self.cfg.effort_limit_scale).clamp_min(
            self.cfg.minimum_effort_limit)
        clipped = torch.clamp(self._joint_efforts, min=-limits, max=limits)
        self._torque_saturated[:] = (~finite) | ((clipped - self._joint_efforts).abs()
                                                 > self.cfg.saturation_tolerance).any(dim=1)
        self._joint_efforts[:] = clipped
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self._current_wrench_s[ids] = 0.0
        self._joint_efforts[ids] = 0.0
        self._torque_saturated[ids] = False
        self._feedback_ready_after[ids] = self._env._sim_step_counter + 1
