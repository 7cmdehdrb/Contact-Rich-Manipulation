"""Bounded, reset-only object placement and palm-facing six-joint IK."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.sim.utils import get_current_stage
import isaaclab.utils.math as math_utils

from ..action_math import inspire_synergy_to_joint_positions, quaternion_rotate
from ..assets.robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from ..collision_geometry import RobotCollisionBounds
from ..contact_math import palm_facing_start_pose


def _ids(env, env_ids):
    return torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(
        env_ids, device=env.device, dtype=torch.long
    )


def reset_cube_on_table(env, env_ids) -> None:
    """Sample the physical Cube once and place its bottom on the thin table."""

    env_ids = _ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    task = env.cfg.task
    cube = env.scene["target_object"]
    state = cube.data.default_root_state[env_ids].clone()
    low = state.new_tensor(task.object_xy_range_low)
    high = state.new_tensor(task.object_xy_range_high)
    state[:, :2] = low + torch.rand((len(env_ids), 2), device=env.device) * (high - low)
    state[:, 2] = task.table_center[2] + .5 * (task.table_size[2] + task.cube_size)
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 3:7] = state.new_tensor((1.0, 0.0, 0.0, 0.0))
    state[:, 7:] = 0.0
    cube.write_root_state_to_sim(state, env_ids=env_ids)


class ContactSafePoseReset(ManagerTermBase):
    """Accept the first valid bounded seed with a single final collision gate.

    Cube position and the sampled hand pose remain fixed across the seeds.
    Joint writes refresh lazy link FK; numerical six-column Jacobians avoid
    stale PhysX Jacobians without forwarding or stepping the physics scene.
    All work is batched over pending reset rows and ends at episode start.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.arm_joint_ids, names = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(names) != ARM_JOINT_NAMES:
            raise RuntimeError(f"Unexpected arm reset joint order: {names}")
        self.hand_joint_ids, names = self.robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        if tuple(names) != HAND_JOINT_NAMES:
            raise RuntimeError(f"Unexpected hand reset joint order: {names}")
        body_ids, names = self.robot.find_bodies(env.cfg.actions.arm_action.body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one reset hand body, found {names}")
        self.hand_body_id = int(body_ids[0])
        self.wrist_local_index = ARM_JOINT_NAMES.index("wrist_3_joint")
        self.c_offset_h = torch.tensor(env.cfg.actions.arm_action.body_offset_pos, device=env.device)
        self.c_quat_h = torch.tensor(env.cfg.actions.arm_action.body_offset_quat, device=env.device)
        robot_root = env.scene.env_prim_paths[0] + "/Robot"
        self.collision_bounds = RobotCollisionBounds(get_current_stage(), robot_root, self.robot.body_names, env.device)
        self.palm_reference_h = self.collision_bounds.palm_face_h
        self.attempts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.accepted_seed_index = torch.full_like(self.attempts, -1)
        self.ik_iterations = torch.zeros_like(self.attempts)
        self.obb_checks = torch.zeros_like(self.attempts)
        self.initial_hand_synergy = torch.zeros((env.num_envs, 2), device=env.device)
        self.desired_c_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
        self.desired_c_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
        self.desired_c_quat_w[:, 0] = 1.0
        self.desired_palm_position_w = torch.zeros_like(self.desired_c_pos_w)
        self.direction_w = torch.zeros_like(self.desired_c_pos_w)
        self.position_error_m = torch.zeros(env.num_envs, device=env.device)
        self.orientation_error_rad = torch.zeros_like(self.position_error_m)
        self.palm_alignment_cos = torch.zeros_like(self.position_error_m)
        self.last_cube_overlap = torch.zeros((env.num_envs, len(self.robot.body_names)), dtype=torch.bool, device=env.device)
        self.last_table_overlap = torch.zeros_like(self.last_cube_overlap)
        self.last_collision_checked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        seed_shape = (env.num_envs, len(env.cfg.task.reset_joint_seed_offsets))
        self.seed_position_error_m = torch.full(seed_shape, torch.nan, device=env.device)
        self.seed_orientation_error_rad = torch.full_like(self.seed_position_error_m, torch.nan)
        self.seed_pose_valid = torch.zeros(seed_shape, dtype=torch.bool, device=env.device)
        self.seed_collision_checked = torch.zeros_like(self.seed_pose_valid)
        self.seed_cube_overlap = torch.zeros((*seed_shape, len(self.robot.body_names)), dtype=torch.bool, device=env.device)
        self.seed_table_overlap = torch.zeros_like(self.seed_cube_overlap)
        # Python call counts distinguish simulator I/O from device-only IK.
        # These cumulative diagnostics never read a GPU scalar during rollout.
        self.reset_io_counters = {name: 0 for name in (
            "joint_state_writes", "joint_position_writes", "physx_pose_queries",
            "software_fk_batches", "software_jacobian_batches",
            "numpy_fk_batches", "packed_host_transfers",
        )}

    def _write_reset_joint_state(self, position, velocity, env_ids):
        self.reset_io_counters["joint_state_writes"] += 1
        self.robot.write_joint_state_to_sim(position, velocity, env_ids=env_ids)

    def _write_reset_joint_position(self, position, env_ids):
        self.reset_io_counters["joint_position_writes"] += 1
        self.robot.write_joint_position_to_sim(position, env_ids=env_ids)

    def __call__(self, env, env_ids) -> None:
        env_ids = _ids(env, env_ids)
        if env_ids.numel() == 0:
            return
        task = env.cfg.task
        for counter in (self.attempts, self.ik_iterations, self.obb_checks):
            counter[env_ids] = 0
        self.accepted_seed_index[env_ids] = -1
        self.position_error_m[env_ids] = torch.inf
        self.orientation_error_rad[env_ids] = torch.inf
        self.palm_alignment_cos[env_ids] = 0.0
        self.last_cube_overlap[env_ids] = False
        self.last_table_overlap[env_ids] = False
        self.last_collision_checked[env_ids] = False
        self.seed_position_error_m[env_ids] = torch.nan
        self.seed_orientation_error_rad[env_ids] = torch.nan
        for history in (self.seed_pose_valid, self.seed_collision_checked, self.seed_cube_overlap, self.seed_table_overlap):
            history[env_ids] = False
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += env.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=env_ids)

        # Each hand coordinate is independent; 0 means closed and 1 open.
        hand_synergy = torch.empty((len(env_ids), 2), device=env.device).uniform_(*task.initial_hand_open_range)
        self.initial_hand_synergy[env_ids] = hand_synergy
        hand_targets = inspire_synergy_to_joint_positions(hand_synergy)
        self._sample_pose(env_ids)
        success = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        accepted_q = self.robot.data.default_joint_pos[env_ids].clone()
        accepted_q[:, self.hand_joint_ids] = hand_targets
        seeds = accepted_q.new_tensor(task.reset_joint_seed_offsets)
        if seeds.ndim != 2 or seeds.shape[1] != 6 or not len(seeds):
            raise ValueError("Reset requires a nonempty bounded list of six-joint seeds")
        for seed_index, seed_offset in enumerate(seeds):
            rows = torch.where(~success)[0]
            if rows.numel() == 0:
                break
            pending_ids = env_ids[rows]
            # A nonconvergent later seed must not inherit an earlier seed's
            # collision/alignment diagnostics.
            self.last_collision_checked[pending_ids] = False
            self.last_cube_overlap[pending_ids] = False
            self.last_table_overlap[pending_ids] = False
            self.palm_alignment_cos[pending_ids] = torch.nan
            self.attempts[pending_ids] += 1
            q = self.robot.data.default_joint_pos[pending_ids].clone()
            q[:, self.arm_joint_ids] += seed_offset
            q[:, self.hand_joint_ids] = hand_targets[rows]
            q[:, self.arm_joint_ids] = self._bounded_arm(q[:, self.arm_joint_ids], pending_ids)
            self._write_reset_joint_state(q, torch.zeros_like(q), pending_ids)
            converged = self._solve_seed(pending_ids, hand_targets[rows])
            self.seed_position_error_m[pending_ids, seed_index] = self.position_error_m[pending_ids]
            self.seed_orientation_error_rad[pending_ids, seed_index] = self.orientation_error_rad[pending_ids]
            if not bool(converged.any()):
                continue
            candidate_rows = rows[converged]
            candidate_ids = env_ids[candidate_rows]
            # Collision certification is performed only on converged states.
            self.obb_checks[candidate_ids] += 1
            pose_valid = self._check_pose_and_limits(candidate_ids)
            collision_free = self._collision_free(candidate_ids)
            self.last_collision_checked[candidate_ids] = True
            self.seed_pose_valid[candidate_ids, seed_index] = pose_valid
            self.seed_collision_checked[candidate_ids, seed_index] = True
            self.seed_cube_overlap[candidate_ids, seed_index] = self.last_cube_overlap[candidate_ids]
            self.seed_table_overlap[candidate_ids, seed_index] = self.last_table_overlap[candidate_ids]
            accepted = pose_valid & collision_free
            if bool(accepted.any()):
                accepted_rows = candidate_rows[accepted]
                accepted_ids = candidate_ids[accepted]
                accepted_q[accepted_rows] = self.robot.data.joint_pos[accepted_ids]
                success[accepted_rows] = True
                self.accepted_seed_index[accepted_ids] = seed_index
        if not bool(success.all()):
            bad = env_ids[~success]
            details = [
                {"env_id": int(i), "attempts": int(self.attempts[i]),
                 "position_error_m": float(self.position_error_m[i]),
                 "orientation_error_rad": float(self.orientation_error_rad[i]),
                 "palm_alignment_cos": float(self.palm_alignment_cos[i]),
                 "collision_checked": bool(self.last_collision_checked[i]),
                 "cube_overlap_bodies": [self.robot.body_names[j] for j in torch.where(self.last_cube_overlap[i])[0].tolist()],
                 "table_overlap_bodies": [self.robot.body_names[j] for j in torch.where(self.last_table_overlap[i])[0].tolist()],
                 "cube_position_local_m": (env.scene["target_object"].data.root_pos_w[i] - env.scene.env_origins[i]).tolist(),
                 "desired_c_position_local_m": (self.desired_c_pos_w[i] - env.scene.env_origins[i]).tolist(),
                 "desired_c_quaternion_wxyz": self.desired_c_quat_w[i].tolist(),
                 "hand_synergy": self.initial_hand_synergy[i].tolist(),
                 "seeds": self.seed_diagnostics(i)}
                for i in bad[:8].tolist()
            ]
            raise RuntimeError(f"Palm-facing reset failed for {len(bad)}/{len(env_ids)} rows; no episode started: {details}")
        zero = torch.zeros_like(accepted_q)
        self._write_reset_joint_state(accepted_q, zero, env_ids)
        self.robot.set_joint_position_target(accepted_q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(zero, env_ids=env_ids)
        self.robot.set_joint_effort_target(zero, env_ids=env_ids)

    def seed_diagnostics(self, env_id):
        """Report only matching seed results; host reads occur on request/failure."""

        return [
            {"seed_index": seed,
             "position_error_m": float(self.seed_position_error_m[env_id, seed]),
             "orientation_error_rad": float(self.seed_orientation_error_rad[env_id, seed]),
             "pose_valid": bool(self.seed_pose_valid[env_id, seed]),
             "collision_checked": bool(self.seed_collision_checked[env_id, seed]),
             "cube_overlap_bodies": [self.robot.body_names[j] for j in torch.where(self.seed_cube_overlap[env_id, seed])[0].tolist()],
             "table_overlap_bodies": [self.robot.body_names[j] for j in torch.where(self.seed_table_overlap[env_id, seed])[0].tolist()]}
            for seed in range(int(self.attempts[env_id]))
        ]

    def _sample_pose(self, env_ids):
        task = self._env.cfg.task
        cube_position = self._env.scene["target_object"].data.root_pos_w[env_ids]
        count = len(env_ids)
        jitter = cube_position.new_tensor(task.reset_position_jitter_task)
        offset = cube_position.new_tensor(task.reset_position_offset_task) + (2.0 * torch.rand((count, 3), device=self._env.device) - 1.0) * jitter
        candidates = []
        eligible = []
        for sign in (-1.0, 1.0):
            candidate = palm_facing_start_pose(cube_position, cube_position.new_full((count,), sign), offset, self.palm_reference_h, self.c_offset_h)
            candidates.append(candidate)
            local_y = candidate[0][:, 1] - self._env.scene.env_origins[env_ids, 1]
            eligible.append((local_y >= task.reset_c_y_range[0]) & (local_y <= task.reset_c_y_range[1]))
        if not bool((eligible[0] | eligible[1]).all()):
            raise RuntimeError("The sampled Cube has no palm-facing reset side inside reset_c_y_range")
        positive = eligible[1] & (~eligible[0] | (torch.rand(count, device=self._env.device) >= .5))
        self.direction_w[env_ids] = 0.0
        self.direction_w[env_ids, 1] = torch.where(positive, 1.0, -1.0)
        self.desired_c_pos_w[env_ids] = torch.where(positive[:, None], candidates[1][0], candidates[0][0])
        hand_rotation = torch.where(positive[:, None, None], candidates[1][1], candidates[0][1])
        hand_quat = math_utils.quat_from_matrix(hand_rotation)
        self.desired_c_quat_w[env_ids] = math_utils.quat_unique(math_utils.quat_mul(hand_quat, self.c_quat_h.expand(count, -1)))
        self.desired_palm_position_w[env_ids] = torch.where(positive[:, None], candidates[1][2], candidates[0][2])

    def _control_pose(self, env_ids):
        self.reset_io_counters["physx_pose_queries"] += 1
        hand_position = self.robot.data.body_link_pos_w[env_ids, self.hand_body_id]
        hand_quaternion = self.robot.data.body_link_quat_w[env_ids, self.hand_body_id]
        return math_utils.combine_frame_transforms(
            hand_position, hand_quaternion, self.c_offset_h.expand(len(env_ids), -1), self.c_quat_h.expand(len(env_ids), -1)
        )

    def palm_reference_pose_w(self, env_ids=None):
        """Return (exposed pad face position, H quaternion); its normal is H +Y."""

        env_ids = _ids(self._env, env_ids)
        hand_position = self.robot.data.body_link_pos_w[env_ids, self.hand_body_id]
        hand_quaternion = self.robot.data.body_link_quat_w[env_ids, self.hand_body_id]
        return hand_position + quaternion_rotate(hand_quaternion, self.palm_reference_h.expand(len(env_ids), -1)), hand_quaternion

    def _pose_errors(self, env_ids):
        position, quaternion = self._control_pose(env_ids)
        return math_utils.compute_pose_error(position, quaternion, self.desired_c_pos_w[env_ids], self.desired_c_quat_w[env_ids], rot_error_type="axis_angle")

    def _bounded_arm(self, arm_q, env_ids):
        task = self._env.cfg.task
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_ids]
        result = torch.clamp(arm_q, limits[..., 0] + task.reset_joint_limit_margin_rad, limits[..., 1] - task.reset_joint_limit_margin_rad)
        result[:, self.wrist_local_index].clamp_(*task.reset_wrist_3_range_rad)
        return result

    def _solve_seed(self, env_ids, hand_targets):
        task = self._env.cfg.task
        converged = torch.zeros(len(env_ids), dtype=torch.bool, device=self._env.device)
        identity = torch.eye(6, device=self._env.device).unsqueeze(0)
        for _ in range(task.reset_max_iterations):
            rows = torch.where(~converged)[0]
            if rows.numel() == 0:
                break
            active_ids = env_ids[rows]
            self.ik_iterations[active_ids] += 1
            position_error, rotation_error = self._pose_errors(active_ids)
            reached = (position_error.norm(dim=-1) <= task.reset_position_tolerance_m) & (rotation_error.norm(dim=-1) <= task.reset_orientation_tolerance_rad)
            converged[rows[reached]] = True
            solve_rows = rows[~reached]
            if solve_rows.numel() == 0:
                break
            solve_ids = env_ids[solve_rows]
            pos_error, rot_error = position_error[~reached], rotation_error[~reached]
            pos_error *= (task.reset_position_error_step_m / pos_error.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)).clamp(max=1.0)
            rot_error *= (task.reset_orientation_error_step_rad / rot_error.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)).clamp(max=1.0)
            error = torch.cat((pos_error, rot_error), dim=-1)
            jacobian = self._numerical_control_jacobian(solve_ids)
            dls = torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + task.reset_damping**2 * identity, error.unsqueeze(-1))
            delta = (task.reset_step_size * (jacobian.transpose(1, 2) @ dls).squeeze(-1)).clamp(-task.reset_joint_delta_limit_rad, task.reset_joint_delta_limit_rad)
            q = self.robot.data.joint_pos[solve_ids].clone()
            q[:, self.arm_joint_ids] = self._bounded_arm(q[:, self.arm_joint_ids] + delta, solve_ids)
            q[:, self.hand_joint_ids] = hand_targets[solve_rows]
            self._write_reset_joint_state(q, torch.zeros_like(q), solve_ids)
        position_error, rotation_error = self._pose_errors(env_ids)
        self.position_error_m[env_ids] = position_error.norm(dim=-1)
        self.orientation_error_rad[env_ids] = rotation_error.norm(dim=-1)
        return (self.position_error_m[env_ids] <= task.reset_position_tolerance_m) & (self.orientation_error_rad[env_ids] <= task.reset_orientation_tolerance_rad)

    def _numerical_control_jacobian(self, env_ids, epsilon=1.0e-3):
        base_q = self.robot.data.joint_pos[env_ids].clone()
        base_pos, base_quat = self._control_pose(env_ids)
        columns = []
        for joint_id in self.arm_joint_ids:
            q = base_q.clone()
            q[:, joint_id] += epsilon
            self._write_reset_joint_position(q, env_ids)
            pos, quat = self._control_pose(env_ids)
            _, angular_delta = math_utils.compute_pose_error(base_pos, base_quat, base_pos, quat, rot_error_type="axis_angle")
            columns.append(torch.cat(((pos - base_pos) / epsilon, angular_delta / epsilon), dim=-1))
        self._write_reset_joint_position(base_q, env_ids)
        return torch.stack(columns, dim=-1)

    def _finger_outward_cos(self, env_ids, hand_quaternion=None):
        """Measure actual H +Z against the horizontal base-to-table direction."""

        if hand_quaternion is None:
            _, hand_quaternion = self.palm_reference_pose_w(env_ids)
        outward = (
            self._env.scene["table"].data.root_pos_w[env_ids]
            - self.robot.data.root_link_pos_w[env_ids]
        )
        outward[:, 2] = 0.0
        outward = torch.nn.functional.normalize(outward, dim=-1)
        finger = quaternion_rotate(
            hand_quaternion, outward.new_tensor((0.0, 0.0, 1.0)).expand(len(env_ids), -1)
        )
        return (finger * outward).sum(dim=-1)

    def _check_pose_and_limits(self, env_ids):
        task = self._env.cfg.task
        position_error, rotation_error = self._pose_errors(env_ids)
        self.position_error_m[env_ids] = position_error.norm(dim=-1)
        self.orientation_error_rad[env_ids] = rotation_error.norm(dim=-1)
        palm, hand_quat = self.palm_reference_pose_w(env_ids)
        normal = quaternion_rotate(hand_quat, palm.new_tensor((0.0, 1.0, 0.0)).expand(len(env_ids), -1))
        to_cube = torch.nn.functional.normalize(self._env.scene["target_object"].data.root_pos_w[env_ids] - palm, dim=-1)
        alignment = (normal * to_cube).sum(dim=-1)
        self.palm_alignment_cos[env_ids] = alignment
        q = self.robot.data.joint_pos[env_ids][:, self.arm_joint_ids]
        limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_ids]
        inside = ((q >= limits[..., 0] + task.reset_joint_limit_margin_rad) & (q <= limits[..., 1] - task.reset_joint_limit_margin_rad)).all(-1)
        wrist = q[:, self.wrist_local_index]
        return (inside & torch.isfinite(q).all(-1)
                & (wrist >= task.reset_wrist_3_range_rad[0]) & (wrist <= task.reset_wrist_3_range_rad[1])
                & (self.position_error_m[env_ids] <= task.reset_position_tolerance_m)
                & (self.orientation_error_rad[env_ids] <= task.reset_orientation_tolerance_rad)
                & (alignment >= math.cos(task.reset_orientation_tolerance_rad))
                & (self._finger_outward_cos(env_ids, hand_quat) > 0.0))

    def _collision_free(self, env_ids):
        task = self._env.cfg.task
        body_pos = self.robot.data.body_link_pos_w[env_ids]
        body_quat = self.robot.data.body_link_quat_w[env_ids]
        cube, table = self._env.scene["target_object"], self._env.scene["table"]
        cube_overlap = self.collision_bounds.overlaps(body_pos, body_quat, cube.data.root_pos_w[env_ids], cube.data.root_quat_w[env_ids], (task.cube_size,) * 3, task.reset_clearance_margin_m)
        table_overlap = self.collision_bounds.overlaps(body_pos, body_quat, table.data.root_pos_w[env_ids], table.data.root_quat_w[env_ids], task.table_size, task.reset_clearance_margin_m)
        self.last_cube_overlap[env_ids], self.last_table_overlap[env_ids] = cube_overlap, table_overlap
        return ~(cube_overlap.any(-1) | table_overlap.any(-1))

    def check_final(self, env_ids=None):
        """Check final FK/limits/palm facing and OBB separation without counters."""

        env_ids = _ids(self._env, env_ids)
        return self._check_pose_and_limits(env_ids) & self._collision_free(env_ids)


__all__ = ["reset_cube_on_table", "ContactSafePoseReset"]
