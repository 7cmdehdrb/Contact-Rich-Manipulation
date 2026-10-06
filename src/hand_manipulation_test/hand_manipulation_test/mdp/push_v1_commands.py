"""Push-v1 contact-height shaping and physical diagnostic histories."""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from isaaclab.utils import configclass

from ..action_math import quaternion_conjugate, quaternion_multiply, quaternion_rotate
from ..geometry import base_link_pose_w, control_point_pose_w
from ..height_math import eef_height_guard
from ..push_state import PushStateTracker
from .push_commands import CubePushCommand, CubePushCommandCfg


_DIAGNOSTIC_NAMES = (
    "raw_palm_contact", "raw_contact_seen", "raw_palm_force_peak_n",
    "side_alignment_gate", "side_gate_seen", "grounded", "grounded_seen",
    "palm_normal_alignment_cos", "palm_force_alignment_cos", "preferred_roll_cos",
    "palm_height_error_m", "approach_gap_m", "actual_forward_m", "best_forward_m",
    "actual_cube_displacement_m", "input_action_clip_fraction", "input_action_clip_any",
    "invalid_input_action", "raw_contact_time_s", "valid_push_contact_time_s",
    "eef_height_above_table_m", "eef_height_failure", "finger_outward_cos",
    "finger_inward_failure", "wrist_2_angle_rad", "wrist_branch_failure",
    "table_failure", "footprint_failure", "fall_failure", "invalid_state_failure",
)


class CubePushV1Command(CubePushCommand):
    """Use a contact target height independently of the collision-safe reset.

    Commands optionally sample uniform angles and lengths around a fixed
    path midpoint, then capture realized reset poses. Approach shaping uses
    a separate contact height; diagnostics and reset-selected roll share the
    memoized physical transition. Progress and success retain their original
    contact, alignment, support, and failure gates.
    """

    cfg: "CubePushV1CommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        height = env.cfg.task.contact_palm_height_m
        if not math.isfinite(height) or height < 0:
            raise ValueError("contact_palm_height_m must be finite and nonnegative")
        self._tracker = PushStateTracker(
            replace(self._tracker.cfg, palm_height_offset_m=height),
            self.num_envs, self.device, dtype=self._tracker.dtype,
        )
        self._latest_snapshot = None
        self._diagnostic_counter = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._best_forward = torch.zeros(self.num_envs, device=self.device)
        self._raw_contact_seen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._side_gate_seen = torch.zeros_like(self._raw_contact_seen)
        self._grounded_seen = torch.zeros_like(self._raw_contact_seen)
        self._raw_contact_time = torch.zeros_like(self._best_forward)
        self._valid_push_time = torch.zeros_like(self._best_forward)
        self._preferred_finger_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._preferred_finger_w[:, 2] = 1.0
        self._roll_penalty = torch.zeros_like(self._best_forward)
        self._contact_distance_penalty = torch.zeros_like(self._best_forward)
        self._eef_height_penalty = torch.zeros_like(self._best_forward)
        self._eef_height = torch.zeros_like(self._best_forward)
        self._height_failure = torch.zeros_like(self._raw_contact_seen)
        self._finger_outward_cos = torch.zeros_like(self._best_forward)
        self._finger_failure = torch.zeros_like(self._raw_contact_seen)
        self._wrist_angle = torch.zeros_like(self._best_forward)
        self._wrist_failure = torch.zeros_like(self._raw_contact_seen)
        task = env.cfg.task
        self._soft_height = getattr(task, "eef_soft_height_above_table_m", .15)
        self._hard_height = getattr(task, "eef_hard_height_above_table_m", .25)
        self._enforce_outward = getattr(task, "enforce_outward_fingers", False)
        self._minimum_outward_cos = getattr(task, "minimum_finger_outward_cos", .25)
        self._wrist_sin_margin = getattr(task, "wrist_2_branch_sin_margin", .15)
        self._wrist_joint_id = None
        if self._enforce_outward:
            ids, _ = env.scene[cfg.asset_name].find_joints("wrist_2_joint", preserve_order=True)
            if len(ids) != 1:
                raise ValueError("Push-v1 requires exactly one wrist_2_joint")
            self._wrist_joint_id = ids[0]
        for name in _DIAGNOSTIC_NAMES:
            self.metrics[name] = torch.zeros_like(self._best_forward)

    @property
    def preferred_finger_w(self):
        """Episode-fixed H +Z direction selected by the actual reset geometry."""

        return self._preferred_finger_w

    @property
    def roll_penalty(self):
        """Cached, nonpositive roll penalty; call state() before consuming it."""

        return self._roll_penalty

    @property
    def contact_distance_penalty(self):
        return self._contact_distance_penalty

    @property
    def eef_height_penalty(self):
        return self._eef_height_penalty

    def _check_centered_sampling(self, env_ids):
        """Reject incompatible geometry before drawing angles or lengths.

        For either side, a direction component is A*sin(angle)+B*cos(angle).
        Its maximum absolute value on the configured jitter interval occurs
        at an endpoint or an interior extremum. This bounds every start and
        goal while preserving the requested uniform command distribution.
        """

        task = self._env.cfg.task
        low, high = task.object_xy_range_low, task.object_xy_range_high
        midpoint = task.command_midpoint_x_offset_m
        jitter = task.command_initial_x_jitter_m
        angle_limit = task.command_angle_jitter_rad
        length_low, length_high = task.command_distance_range
        values = (*low, *high, *task.table_center, *task.table_size, task.cube_size,
                  task.table_path_margin_m, midpoint, jitter, angle_limit, length_low, length_high)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Centered Push sampling configuration must be finite")
        if (jitter < 0 or not 0 <= angle_limit < math.pi / 2 or not 0 < length_low <= length_high
                or any(low[axis] > high[axis] for axis in range(2)) or task.cube_size <= 0
                or any(size <= 0 for size in task.table_size) or task.table_path_margin_m < 0):
            raise ValueError("Centered Push sampling ranges and jitter are invalid")
        count = len(env_ids)
        sine_basis = self._directions(torch.full((count,), math.pi / 2, device=self.device), env_ids)
        cosine_basis = self._directions(torch.zeros(count, device=self.device), env_ids)
        endpoint_low = -sine_basis * math.sin(angle_limit) + cosine_basis * math.cos(angle_limit)
        endpoint_high = sine_basis * math.sin(angle_limit) + cosine_basis * math.cos(angle_limit)
        peak_angle = torch.remainder(torch.atan2(sine_basis, cosine_basis) + math.pi / 2, math.pi) - math.pi / 2
        radius = torch.sqrt(sine_basis.square() + cosine_basis.square())
        bound = torch.where(peak_angle.abs() <= angle_limit + 1.e-7, radius,
                            torch.maximum(endpoint_low.abs(), endpoint_high.abs()))
        midpoint_x = task.table_center[0] + midpoint
        span_x = .5 * length_high * bound[:, 0] + jitter
        padding = math.sqrt(3.) * task.cube_size / 2 + task.table_path_margin_m
        safe_x = task.table_size[0] / 2 - padding
        safe_y = task.table_size[1] / 2 - padding
        valid = (midpoint_x - span_x >= low[0] - 1.e-7) & (midpoint_x + span_x <= high[0] + 1.e-7)
        valid &= abs(midpoint) + span_x <= safe_x + 1.e-7
        valid &= low[1] - length_high * bound[:, 1] >= task.table_center[1] - safe_y - 1.e-7
        valid &= high[1] + length_high * bound[:, 1] <= task.table_center[1] + safe_y + 1.e-7
        valid &= bound[:, 2] <= 1.e-6
        if not bool(valid.all()):
            raise ValueError("Centered Push sampling bounds must contain every configured angle/length and complete Table path")

    def sample_reset(self, env_ids):
        """Draw angle and length once, then center each complete X path.

        One-shot fixed specifications keep the parent's validation contract.
        Random starts use midpoint_x - 0.5*length*direction_x plus independent
        X jitter. Invalid settings raise instead of rejecting sampled angles
        or lengths, so command frequencies remain uniform.
        """

        task = self._env.cfg.task
        if not getattr(task, "command_centered_path", False):
            return super().sample_reset(env_ids)
        index = self._ids(env_ids)
        if not len(index):
            return self.sampled_cube_pos_w[index]
        fixed = self._specified[index]
        random_rows = index[~fixed]
        position = self._specified_position[index].clone()
        angle = self._specified_angle[index].clone()
        distance = self._specified_distance[index].clone()
        if len(random_rows):
            self._check_centered_sampling(random_rows)
            count = len(random_rows)
            sampled_angle = (2. * torch.rand(count, device=self.device) - 1.) * task.command_angle_jitter_rad
            sampled_angle += torch.randint(0, 2, (count,), device=self.device) * math.pi
            sampled_distance = torch.empty(count, device=self.device).uniform_(*task.command_distance_range)
            sampled_direction = self._directions(sampled_angle, random_rows)
            sampled_position = torch.empty((count, 3), device=self.device)
            sampled_position[:, 0] = (
                task.table_center[0] + task.command_midpoint_x_offset_m
                - .5 * sampled_distance * sampled_direction[:, 0]
                + (2. * torch.rand(count, device=self.device) - 1.) * task.command_initial_x_jitter_m
            )
            low_y, high_y = task.object_xy_range_low[1], task.object_xy_range_high[1]
            sampled_position[:, 1] = low_y + torch.rand(count, device=self.device) * (high_y - low_y)
            sampled_position[:, 2] = task.cube_center_height_m
            sampled_position += self._env.scene.env_origins[random_rows]
            position[~fixed], angle[~fixed], distance[~fixed] = sampled_position, sampled_angle, sampled_distance
        direction = self._directions(angle, index)
        origins = self._env.scene.env_origins[index]
        local_position = position - origins
        # Adding an env-local start to a large float32 origin can round its
        # boundary by a world-coordinate ULP. Honor that representation error
        # when converting back to local coordinates, while leaving the actual
        # conservative Table-path and physical failure checks unchanged.
        magnitude = torch.maximum(position.abs(), origins.abs())
        spacing = torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))) - magnitude
        tolerance = spacing.clamp_min(1.e-7)
        low, high = position.new_tensor(task.object_xy_range_low), position.new_tensor(task.object_xy_range_high)
        valid = torch.isfinite(position).all(-1) & torch.isfinite(angle) & torch.isfinite(distance)
        valid &= ((local_position[:, :2] >= low - tolerance[:, :2]) &
                  (local_position[:, :2] <= high + tolerance[:, :2])).all(-1)
        valid &= (local_position[:, 2] - task.cube_center_height_m).abs() <= tolerance[:, 2].clamp_min(1.e-6)
        valid &= self._valid_path(position, direction, distance, index)
        if not bool(valid.all()):
            raise RuntimeError("Centered Push draw violates Cube bounds or complete Table clearance; no commands were committed")
        self.sampled_cube_pos_w[index] = position
        self.angle_rad[index] = angle
        self.distance_m[index] = distance
        self.direction_w[index] = direction
        self.target_pos_w[index] = position
        self.goal_pos_w[index] = position + distance[:, None] * direction
        self._pending[index] = True
        self._specified[index] = False
        return self.sampled_cube_pos_w[index]

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        index = self._ids(env_ids)
        safe = self._env.event_manager.get_term_cfg("safe_hand").func
        # C = H * C_in_H. Recover the chosen H frame rather than imposing a
        # global finger direction that can contradict a tilted reset branch.
        hand_quaternion = quaternion_multiply(
            safe.desired_c_quat_w[index], quaternion_conjugate(safe.c_quat_h)
        )
        finger_z = hand_quaternion.new_tensor((0., 0., 1.)).expand(len(index), -1)
        self._preferred_finger_w[index] = quaternion_rotate(hand_quaternion, finger_z)
        self._diagnostic_counter[index] = -1
        self._best_forward[index] = 0
        self._raw_contact_seen[index] = False
        self._side_gate_seen[index] = False
        self._grounded_seen[index] = False
        self._raw_contact_time[index] = 0
        self._valid_push_time[index] = 0
        penalty = self._roll_penalty.clone()
        penalty[index] = 0
        self._roll_penalty = penalty
        self._contact_distance_penalty[index] = 0
        self._eef_height_penalty[index] = 0
        for name in _DIAGNOSTIC_NAMES:
            self.metrics[name][index] = 0

    def _snapshot(self):
        sample = super()._snapshot()
        safe = self._env.event_manager.get_term_cfg("safe_hand").func
        reference = getattr(safe, "contact_reference_pose_w", None)
        if reference is not None:
            contact_position, hand_quaternion = reference()
            direction = torch.nn.functional.normalize(sample.direction_w, dim=-1, eps=1.e-8)
            cube_direction = quaternion_rotate(quaternion_conjugate(sample.cube_quat_w), direction)
            extent = cube_direction.abs().sum(-1) * self._tracker.cfg.cube_size_m / 2
            target = sample.cube_pos_w - extent[:, None] * direction
            target = target.clone()
            target[:, 2] += self._tracker.cfg.palm_height_offset_m
            sample = replace(sample, palm_pos_w=contact_position, hand_quat_w=hand_quaternion,
                             approach_target_pos_w=target)
        control_position, _ = control_point_pose_w(self._env)
        height, penalty, height_failure = eef_height_guard(
            control_position, sample.table_pos_w, sample.table_quat_w,
            table_thickness_m=self._env.cfg.task.table_size[2],
            soft_height_m=self._soft_height, hard_height_m=self._hard_height,
        )
        live = sample.live.bool()
        self._eef_height = height
        self._eef_height_penalty = penalty * live.to(penalty.dtype)
        self._height_failure = height_failure & live
        if self._enforce_outward:
            base_position, _ = base_link_pose_w(self._env)
            outward = sample.initial_cube_pos_w - base_position
            outward = outward.clone()
            outward[:, 2] = 0
            outward = torch.nn.functional.normalize(outward, dim=-1, eps=1.e-8)
            finger = quaternion_rotate(sample.hand_quat_w, outward.new_tensor((0., 0., 1.)).expand_as(outward))
            self._finger_outward_cos = torch.nan_to_num((finger * outward).sum(-1), nan=-1.).clamp(-1., 1.)
            self._finger_failure = (self._finger_outward_cos < self._minimum_outward_cos) & live
            wrist = self._env.scene[self.cfg.asset_name].data.joint_pos[:, self._wrist_joint_id]
            self._wrist_angle = torch.atan2(wrist.sin(), wrist.cos())
            # The physical branch is periodic. A 2*pi representation change
            # is harmless; crossing the 0/pi wrist singularity is not.
            self._wrist_failure = (~torch.isfinite(wrist) | (wrist.sin() <= self._wrist_sin_margin)) & live
        self._latest_snapshot = replace(sample, additional_failure=(
            self._height_failure | self._finger_failure | self._wrist_failure
        ))
        return self._latest_snapshot

    def _record_metrics(self, state):
        super()._record_metrics(state)
        sample = self._latest_snapshot
        counter = self._env._sim_step_counter
        changed = self._diagnostic_counter != counter
        live = sample.live.bool()
        active = changed & live
        direction = torch.nan_to_num(sample.direction_w)
        direction = torch.nn.functional.normalize(direction, dim=-1, eps=1.e-8)
        forces = torch.nan_to_num(sample.cube_palm_forces_w_history[:, :, 0])
        norm = torch.linalg.vector_norm(forces, dim=-1)
        pair_active = norm >= self._tracker.cfg.contact_threshold_n
        force_cos = (forces * direction[:, None, None]).sum(-1) / norm.clamp_min(1.e-8)
        side_gate = (pair_active & (force_cos >= self._tracker.cfg.alignment_cos)).flatten(1).any(-1)
        side_gate &= (state.palm_alignment_cos >= self._tracker.cfg.alignment_cos) & live
        raw_contact = state.palm_cube_contact & live
        forward = ((sample.cube_pos_w - sample.initial_cube_pos_w) * direction).sum(-1)
        forward = torch.nan_to_num(forward, nan=0., posinf=0., neginf=0.)
        self._best_forward = torch.where(active, torch.maximum(self._best_forward, forward), self._best_forward)
        self._raw_contact_seen |= active & raw_contact
        self._side_gate_seen |= active & side_gate
        self._grounded_seen |= active & state.grounded
        self._raw_contact_time += (active & raw_contact).float() * self._env.step_dt
        self._valid_push_time += (active & state.valid_push_contact & state.grounded).float() * self._env.step_dt

        # Rotation about the desired normal is measured after projection onto
        # its perpendicular plane. A normal mismatch is penalized separately.
        actual_finger = quaternion_rotate(
            sample.hand_quat_w, direction.new_tensor((0., 0., 1.)).expand_as(direction)
        )
        preferred = self._preferred_finger_w
        actual = actual_finger - (actual_finger * direction).sum(-1, keepdim=True) * direction
        target = preferred - (preferred * direction).sum(-1, keepdim=True) * direction
        actual = torch.nn.functional.normalize(actual, dim=-1, eps=1.e-8)
        target = torch.nn.functional.normalize(target, dim=-1, eps=1.e-8)
        roll_cos = torch.nan_to_num((actual * target).sum(-1), nan=0., posinf=0., neginf=0.).clamp(-1., 1.)
        roll_penalty = -.5 * (1. - roll_cos) * (live & ~state.failure).float()
        self._roll_penalty = torch.where(changed, roll_penalty, self._roll_penalty)

        cube_direction = quaternion_rotate(quaternion_conjugate(sample.cube_quat_w), direction)
        cube_extent = cube_direction.abs().sum(-1) * self._tracker.cfg.cube_size_m / 2
        approach_position = sample.cube_pos_w - cube_extent[:, None] * direction
        approach_position = approach_position.clone()
        approach_position[:, 2] += self._tracker.cfg.palm_height_offset_m
        if sample.approach_target_pos_w is not None:
            approach_position = sample.approach_target_pos_w
        height_error = sample.palm_pos_w[:, 2] - approach_position[:, 2]
        gap = torch.linalg.vector_norm(sample.palm_pos_w - approach_position, dim=-1)
        proximity_cost = -torch.nan_to_num(gap, nan=0., posinf=0., neginf=0.) * (live & ~state.failure).to(gap.dtype)
        self._contact_distance_penalty = torch.where(changed, proximity_cost, self._contact_distance_penalty)

        # ActionManager preserves the caller's unsanitized input. Individual
        # action terms expose already-clipped values, so those cannot measure
        # input clipping. These diagnostics are distinct from torque saturation.
        manager = getattr(self._env, "action_manager", None)
        actions = getattr(manager, "action", None)
        if actions is None:
            clip_fraction = torch.zeros_like(forward)
            clip_any = torch.zeros_like(live)
            invalid_action = torch.zeros_like(live)
        else:
            finite = torch.isfinite(actions)
            clipped = finite & (actions.abs() > 1.)
            clip_fraction = clipped.float().mean(-1)
            clip_any = clipped.any(-1)
            invalid_action = ~finite.all(-1)
        instantaneous = {
            "raw_palm_contact": raw_contact.float(), "raw_palm_force_peak_n": norm.flatten(1).amax(-1),
            "side_alignment_gate": side_gate.float(), "grounded": state.grounded.float(),
            "palm_normal_alignment_cos": state.palm_alignment_cos, "palm_force_alignment_cos": state.force_alignment_cos,
            "preferred_roll_cos": roll_cos, "palm_height_error_m": height_error,
            "approach_gap_m": gap, "actual_forward_m": forward,
            "actual_cube_displacement_m": torch.linalg.vector_norm(sample.cube_pos_w - sample.initial_cube_pos_w, dim=-1),
            "input_action_clip_fraction": clip_fraction, "input_action_clip_any": clip_any.float(),
            "invalid_input_action": invalid_action.float(),
            "eef_height_above_table_m": self._eef_height,
            "eef_height_failure": self._height_failure.float(),
            "finger_outward_cos": self._finger_outward_cos,
            "finger_inward_failure": self._finger_failure.float(),
            "wrist_2_angle_rad": self._wrist_angle,
            "wrist_branch_failure": self._wrist_failure.float(),
            "table_failure": state.table_failure.float(),
            "footprint_failure": state.footprint_failure.float(),
            "fall_failure": state.fall_failure.float(),
            "invalid_state_failure": state.invalid_state.float(),
        }
        for name, value in instantaneous.items():
            value = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.) * live.float()
            self.metrics[name][:] = torch.where(changed, value, self.metrics[name])
        histories = {
            "raw_contact_seen": self._raw_contact_seen.float(), "side_gate_seen": self._side_gate_seen.float(),
            "grounded_seen": self._grounded_seen.float(), "best_forward_m": self._best_forward,
            "raw_contact_time_s": self._raw_contact_time, "valid_push_contact_time_s": self._valid_push_time,
        }
        for name, value in histories.items():
            self.metrics[name][:] = value
        self._diagnostic_counter[changed] = counter


@configclass
class CubePushV1CommandCfg(CubePushCommandCfg):
    class_type: type = CubePushV1Command


__all__ = ["CubePushV1Command", "CubePushV1CommandCfg"]
