"""Command-coupled Cube placement and bounded horizontal palm-facing reset."""

from __future__ import annotations

import math

import torch
import numpy as np

import isaaclab.utils.math as math_utils
from isaaclab.sim.utils import get_current_stage

from ..action_math import quaternion_rotate
from ..push_math import push_palm_start_pose
from ..reset_kinematics import (
    NumpySerialArmKinematics, bounded_dls_correction, cached_ik_convergence_tolerances,
    serial_arm_kinematics_from_usd,
)
from .contact_events import ContactSafePoseReset, _ids


def reset_cube_for_push(env, env_ids) -> None:
    """Sample one path specification, then place its Cube exactly once."""

    env_ids = _ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    command = env.command_manager.get_term("target_position")
    command.sample_reset(env_ids)
    cube = env.scene["target_object"]
    state = cube.data.default_root_state[env_ids].clone()
    state[:, :3] = command.sampled_cube_pos_w[env_ids]
    state[:, 3:7] = state.new_tensor((1.0, 0.0, 0.0, 0.0))
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state, env_ids=env_ids)


class PushSafePoseReset(ContactSafePoseReset):
    """Solve pending rows with cached FK, then certify real PhysX poses and OBBs."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.use_cached_kinematics = True
        # Small asynchronous reset groups are launch-bound on CUDA. Retain
        # PhysX for large batches until the profiling API establishes a better
        # backend for that hardware; both paths keep the same real final gate.
        self.reset_backend = "auto"
        self.numpy_reset_max_rows = 32
        self._arm_kinematics = serial_arm_kinematics_from_usd(
            get_current_stage(), env.scene.env_prim_paths[0] + "/Robot",
            env.cfg.actions.arm_action.joint_names, env.cfg.actions.arm_action.body_name,
            env.cfg.actions.arm_action.body_offset_pos, env.cfg.actions.arm_action.body_offset_quat,
            device=env.device,
        )
        self._ik_identity = torch.eye(6, device=env.device).unsqueeze(0)
        self._numpy_arm_kinematics = NumpySerialArmKinematics(self._arm_kinematics)
        self.reset_io_counters["real_fk_refinements"] = 0

    def cached_control_pose_w(self, env_ids=None, joint_positions=None):
        """Expose cached FK for comparison with real simulator link poses."""

        index = _ids(self._env, env_ids)
        q = self.robot.data.joint_pos[index][:, self.arm_joint_ids] if joint_positions is None else joint_positions
        root = self.robot.data.root_link_pose_w[index]
        return self._arm_kinematics.pose(q, root)

    def cached_numerical_control_jacobian(self, env_ids=None):
        index = _ids(self._env, env_ids)
        q = self.robot.data.joint_pos[index][:, self.arm_joint_ids]
        root = self.robot.data.root_link_pose_w[index]
        return self._arm_kinematics.pose_and_jacobian(q, root)[2]

    def _solve_seed(self, env_ids, hand_targets):
        # Retain the original path as a profiling/reference API; environment
        # configuration and normal policy steps do not switch implementations.
        if not self.use_cached_kinematics:
            return super()._solve_seed(env_ids, hand_targets)
        backend = self.reset_backend
        if backend == "auto":
            backend = "numpy" if len(env_ids) <= self.numpy_reset_max_rows else "physx"
        if backend == "physx":
            return super()._solve_seed(env_ids, hand_targets)
        if backend == "numpy":
            return self._solve_seed_numpy(env_ids)
        if backend != "torch":
            raise ValueError(f"Unknown Push reset backend: {backend}")
        return self._solve_seed_torch(env_ids, hand_targets)

    def _solve_seed_numpy(self, env_ids):
        task = self._env.cfg.task
        position_tolerance, orientation_tolerance = cached_ik_convergence_tolerances(
            task.reset_position_tolerance_m, task.reset_orientation_tolerance_rad
        )
        q = self.robot.data.joint_pos[env_ids].clone()
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_ids]
        lower = limits[..., 0] + task.reset_joint_limit_margin_rad
        upper = limits[..., 1] - task.reset_joint_limit_margin_rad
        lower[:, self.wrist_local_index].clamp_(min=task.reset_wrist_3_range_rad[0])
        upper[:, self.wrist_local_index].clamp_(max=task.reset_wrist_3_range_rad[1])
        packed = torch.cat((q[:, self.arm_joint_ids], self.robot.data.root_link_pose_w[env_ids],
                            self.desired_c_pos_w[env_ids], self.desired_c_quat_w[env_ids], lower, upper), dim=-1)
        data = packed.detach().cpu().numpy()
        self.reset_io_counters["packed_host_transfers"] += 1
        solved_q, counts, batches = self._numpy_arm_kinematics.solve(
            data[:, :6], data[:, 6:13], data[:, 13:20], data[:, 20:26], data[:, 26:32],
            max_iterations=task.reset_max_iterations, damping=task.reset_damping, step_size=task.reset_step_size,
            position_error_step=task.reset_position_error_step_m, orientation_error_step=task.reset_orientation_error_step_rad,
            delta_limit=task.reset_joint_delta_limit_rad, position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        self.reset_io_counters["numpy_fk_batches"] += batches
        self.reset_io_counters["software_fk_batches"] += batches
        self.reset_io_counters["software_jacobian_batches"] += batches
        # Pack solver results and per-row instrumentation into one return
        # transfer; all hand states and non-arm joints remain the sampled seed.
        result = torch.as_tensor(np.concatenate((solved_q, counts[:, None]), axis=-1), device=self._env.device, dtype=q.dtype)
        q[:, self.arm_joint_ids] = result[:, :6]
        seed_iterations = result[:, 6].to(dtype=torch.long)
        self.ik_iterations[env_ids] += seed_iterations
        self._write_reset_joint_state(q, torch.zeros_like(q), env_ids)
        return self._finish_cached_seed(env_ids, seed_iterations)

    def _solve_seed_torch(self, env_ids, hand_targets):
        task = self._env.cfg.task
        position_tolerance, orientation_tolerance = cached_ik_convergence_tolerances(
            task.reset_position_tolerance_m, task.reset_orientation_tolerance_rad
        )
        q = self.robot.data.joint_pos[env_ids].clone()
        root = self.robot.data.root_link_pose_w[env_ids]
        converged = torch.zeros(len(env_ids), dtype=torch.bool, device=self._env.device)
        seed_iterations = torch.zeros(len(env_ids), dtype=torch.long, device=self._env.device)
        for _ in range(task.reset_max_iterations):
            rows = torch.where(~converged)[0]
            if rows.numel() == 0:
                break
            active_ids = env_ids[rows]
            self.ik_iterations[active_ids] += 1
            seed_iterations[rows] += 1
            position, quaternion, jacobian = self._arm_kinematics.pose_and_jacobian(
                q[rows][:, self.arm_joint_ids], root[rows]
            )
            self.reset_io_counters["software_fk_batches"] += 1
            self.reset_io_counters["software_jacobian_batches"] += 1
            pos_error, rot_error = math_utils.compute_pose_error(
                position, quaternion, self.desired_c_pos_w[active_ids], self.desired_c_quat_w[active_ids],
                rot_error_type="axis_angle",
            )
            reached = (pos_error.norm(dim=-1) <= position_tolerance) & (
                rot_error.norm(dim=-1) <= orientation_tolerance
            )
            converged[rows[reached]] = True
            solve_rows = rows[~reached]
            if solve_rows.numel() == 0:
                break
            solve_ids = env_ids[solve_rows]
            pos_error, rot_error = pos_error[~reached], rot_error[~reached]
            pos_error *= (task.reset_position_error_step_m / pos_error.norm(dim=-1, keepdim=True).clamp_min(1.e-9)).clamp(max=1.)
            rot_error *= (task.reset_orientation_error_step_rad / rot_error.norm(dim=-1, keepdim=True).clamp_min(1.e-9)).clamp(max=1.)
            error = torch.cat((pos_error, rot_error), dim=-1)
            jacobian = jacobian[~reached]
            dls = torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + task.reset_damping**2 * self._ik_identity,
                                    error.unsqueeze(-1))
            delta = (task.reset_step_size * (jacobian.transpose(1, 2) @ dls).squeeze(-1)).clamp(
                -task.reset_joint_delta_limit_rad, task.reset_joint_delta_limit_rad
            )
            next_q = q[solve_rows].clone()
            next_q[:, self.arm_joint_ids] = self._bounded_arm(next_q[:, self.arm_joint_ids] + delta, solve_ids)
            next_q[:, self.hand_joint_ids] = hand_targets[solve_rows]
            q[solve_rows] = next_q
        # One endpoint write refreshes real link FK for the existing pose,
        # joint-limit and complete robot/Cube/Table collision certification.
        self._write_reset_joint_state(q, torch.zeros_like(q), env_ids)
        return self._finish_cached_seed(env_ids, seed_iterations)

    def _finish_cached_seed(self, env_ids, seed_iterations):
        """Check stored poses; refine at most once within the same seed budget."""

        task = self._env.cfg.task
        position_error, rotation_error = self._pose_errors(env_ids)
        outside = (position_error.norm(dim=-1) > task.reset_position_tolerance_m) | (
            rotation_error.norm(dim=-1) > task.reset_orientation_tolerance_rad
        )
        rows = torch.where(outside & (seed_iterations < task.reset_max_iterations))[0]
        if rows.numel():
            index = env_ids[rows]
            # This rare path uses measured link FK/Jacobians, rather than
            # repeating the same cached estimate after a float32 writeback.
            delta = bounded_dls_correction(
                self._numerical_control_jacobian(index), position_error[rows], rotation_error[rows],
                damping=task.reset_damping, step_size=task.reset_step_size,
                position_error_step=task.reset_position_error_step_m,
                orientation_error_step=task.reset_orientation_error_step_rad,
                delta_limit=task.reset_joint_delta_limit_rad,
            )
            q = self.robot.data.joint_pos[index].clone()
            q[:, self.arm_joint_ids] = self._bounded_arm(q[:, self.arm_joint_ids] + delta, index)
            self._write_reset_joint_state(q, torch.zeros_like(q), index)
            self.ik_iterations[index] += 1
            self.reset_io_counters["real_fk_refinements"] += 1
            position_error, rotation_error = self._pose_errors(env_ids)
        self.position_error_m[env_ids] = position_error.norm(dim=-1)
        self.orientation_error_rad[env_ids] = rotation_error.norm(dim=-1)
        return (self.position_error_m[env_ids] <= task.reset_position_tolerance_m) & (
            self.orientation_error_rad[env_ids] <= task.reset_orientation_tolerance_rad
        )

    def _sample_pose(self, env_ids):
        task = self._env.cfg.task
        cube = self._env.scene["target_object"].data.root_pos_w[env_ids]
        command = self._env.command_manager.get_term("target_position")
        direction = command.direction_w[env_ids]
        jitter = cube.new_tensor(task.reset_position_jitter_task)
        offset = cube.new_tensor(task.reset_position_offset_task) + (
            2.0 * torch.rand((len(env_ids), 3), device=self._env.device) - 1.0
        ) * jitter
        control_position, hand_rotation, palm_position = push_palm_start_pose(
            cube, direction, offset, self.palm_reference_h, self.c_offset_h
        )
        local_y = control_position[:, 1] - self._env.scene.env_origins[env_ids, 1]
        if not bool(((local_y >= task.reset_c_y_range[0]) & (local_y <= task.reset_c_y_range[1])).all()):
            raise RuntimeError("The sampled push path places initial C outside reset_c_y_range")
        self.direction_w[env_ids] = hand_rotation[:, :, 1]
        self.desired_c_pos_w[env_ids] = control_position
        hand_quaternion = math_utils.quat_from_matrix(hand_rotation)
        self.desired_c_quat_w[env_ids] = math_utils.quat_unique(
            math_utils.quat_mul(hand_quaternion, self.c_quat_h.expand(len(env_ids), -1))
        )
        self.desired_palm_position_w[env_ids] = palm_position

    def _check_pose_and_limits(self, env_ids):
        """Check the horizontal command normal, without aiming down at the Cube."""

        task = self._env.cfg.task
        position_error, rotation_error = self._pose_errors(env_ids)
        self.position_error_m[env_ids] = position_error.norm(dim=-1)
        self.orientation_error_rad[env_ids] = rotation_error.norm(dim=-1)
        palm, hand_quaternion = self.palm_reference_pose_w(env_ids)
        normal = quaternion_rotate(hand_quaternion, palm.new_tensor((0.0, 1.0, 0.0)).expand(len(env_ids), -1))
        alignment = (normal * self.direction_w[env_ids]).sum(dim=-1)
        self.palm_alignment_cos[env_ids] = alignment
        q = self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_ids]
        inside = (
            (q >= limits[..., 0] + task.reset_joint_limit_margin_rad)
            & (q <= limits[..., 1] - task.reset_joint_limit_margin_rad)
        ).all(-1)
        wrist = q[:, self.wrist_local_index]
        tolerance = task.reset_orientation_tolerance_rad
        return (
            inside & torch.isfinite(q).all(-1)
            & (wrist >= task.reset_wrist_3_range_rad[0]) & (wrist <= task.reset_wrist_3_range_rad[1])
            & (self.position_error_m[env_ids] <= task.reset_position_tolerance_m)
            & (self.orientation_error_rad[env_ids] <= tolerance)
            & (alignment >= math.cos(tolerance)) & (normal[:, 2].abs() <= math.sin(tolerance))
            & (self._finger_outward_cos(env_ids, hand_quaternion) > 0.0)
        )


__all__ = ["reset_cube_for_push", "PushSafePoseReset"]
