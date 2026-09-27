"""The two reset events used by the blind-sweeping task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ..assets.robot import ARM_JOINT_NAMES, HAND_CLOSED_TARGETS, HAND_JOINT_NAMES, HAND_OPEN_TARGETS

if TYPE_CHECKING:
    from ..env import BlindSweepEnv


def reset_episode_scene(
    env: "BlindSweepEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Full reset event: restore the board/cube and clear episode-local state.

    Episode mode, direction, distance and object start have already been sampled
    once by :meth:`BlindSweepEnv.sample_episode_specs` before the reset event
    manager runs.  This function only applies that stored specification; it
    never samples a second command.
    """

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    target: RigidObject = env.scene[object_cfg.name]
    pose = torch.cat((env.object_initial_pos_w[env_ids], env.object_initial_quat_w[env_ids]), dim=-1)
    velocity = torch.zeros((len(env_ids), 6), device=env.device)
    target.write_root_pose_to_sim(pose, env_ids=env_ids)
    target.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    env.previous_policy_action[env_ids] = 0.0
    env.last_policy_action[env_ids] = 0.0
    env.has_previous_policy_action[env_ids] = False
    env.sensor_valid[env_ids] = False
    env.sensor_data_fresh[env_ids] = False
    env.board_force_peak[env_ids] = 0.0
    env.board_link_force_peak[env_ids] = 0.0
    env.board_link_index_peak[env_ids] = 0
    env.board_link_position_peak_w[env_ids] = 0.0
    env.board_contact_c_position_peak_w[env_ids] = 0.0
    env.tilt_peak[env_ids] = 0.0
    env.height_peak[env_ids] = 0.0
    env.board_hard_latched[env_ids] = False
    env.tilt_hard_latched[env_ids] = False
    env.height_hard_latched[env_ids] = False
    env.invalid_sim_latched[env_ids] = False
    env.invalid_reset_latched[env_ids] = False
    env.reset_collision_free[env_ids] = False
    env.failure_reason[env_ids] = 0
    env.episode_physics_substep_count[env_ids] = 0
    env.first_selected_contact_substep[env_ids] = -1
    env.selected_contact_previous[env_ids] = False
    env.selected_contact_loss_count[env_ids] = 0
    env.arm_torque_saturation_steps[env_ids] = 0
    env.hand_action_saturation_steps[env_ids] = 0
    env.reset_episode_spec_attempts[env_ids] = 0


def _hand_targets_from_synergy(
    common_open: torch.Tensor,
    thumb_open: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Expand two normalized hand coordinates into the twelve real joints."""

    opened = torch.tensor([HAND_OPEN_TARGETS[name] for name in HAND_JOINT_NAMES], device=device)
    closed = torch.tensor([HAND_CLOSED_TARGETS[name] for name in HAND_JOINT_NAMES], device=device)
    common = common_open.clamp(0.0, 1.0).unsqueeze(-1)
    values = (1.0 - common) * closed + common * opened
    thumb_index = HAND_JOINT_NAMES.index("inspire_left_thumb_1_joint")
    thumb = thumb_open.clamp(0.0, 1.0)
    values[:, thumb_index] = (1.0 - thumb) * closed[thumb_index] + thumb * opened[thumb_index]
    return values


class StableOffsetPoseReset(ManagerTermBase):
    """Place C at a conservative target-relative position and orientation.

    This is the active reset term.  It follows Sweep-Policy's reaching-pose
    reset pattern: build a privileged target-relative pose, solve it from a
    small bounded seed set, and keep the first acceptable solution.  Joint
    writes invalidate Isaac Lab's link-pose buffers, so reading the body pose
    performs the required kinematic refresh without advancing or globally
    forwarding the physics scene inside the IK loop.

    The sampled region is intentionally narrow.  Position coordinates are
    ``(backoff, lateral, height)`` in a task basis made from ``-sweep``, the
    shelf tangent, and world up.  Orientation offsets are local hand-frame RPY.
    A solution is exposed only after both the fast clearance proxy and the
    conservative live-FK OBB certificate accept Robot--Cube and Robot--Board
    separation.  :class:`ConditionalPoseIKReset` below is retained for audit
    and comparison, but is no longer referenced by the environment config.
    """

    def __init__(self, cfg: EventTermCfg, env: "BlindSweepEnv"):
        super().__init__(cfg=cfg, env=env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["robot_cfg"].name]
        self.arm_joint_ids, arm_names = self.robot.find_joints(
            list(params["arm_joint_names"]), preserve_order=True
        )
        if tuple(arm_names) != tuple(params["arm_joint_names"]):
            raise RuntimeError(f"Unexpected arm joint order: {arm_names}")
        self.hand_joint_ids, hand_names = self.robot.find_joints(
            list(HAND_JOINT_NAMES), preserve_order=True
        )
        if tuple(hand_names) != HAND_JOINT_NAMES:
            raise RuntimeError(f"Unexpected Inspire joint order: {hand_names}")
        body_ids, body_names = self.robot.find_bodies(
            params["hand_body_name"], preserve_order=True
        )
        if len(body_ids) != 1 or body_names[0] != params["hand_body_name"]:
            raise RuntimeError(
                f"Expected one hand body {params['hand_body_name']!r}, got {body_names}"
            )
        self.hand_body_id = int(body_ids[0])
        self.jacobian_body_id = (
            self.hand_body_id - 1 if self.robot.is_fixed_base else self.hand_body_id
        )
        self.jacobian_joint_ids = (
            self.arm_joint_ids
            if self.robot.is_fixed_base
            else [joint_id + 6 for joint_id in self.arm_joint_ids]
        )
        self.wrist_local_index = arm_names.index("wrist_3_joint")
        self.c_offset_h = torch.as_tensor(
            params["c_offset_h"], dtype=torch.float32, device=env.device
        )
        self.c_quat_h = torch.as_tensor(
            params["c_quat_h"], dtype=torch.float32, device=env.device
        )
        self.seed_offsets = torch.as_tensor(
            params["joint_seed_offsets"], dtype=torch.float32, device=env.device
        )
        if self.seed_offsets.ndim != 2 or self.seed_offsets.shape[1] != 6:
            raise ValueError("Every stable-reset joint seed must contain six arm values")

    def __call__(
        self,
        env: "BlindSweepEnv",
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        arm_joint_names: tuple[str, ...],
        hand_body_name: str,
        c_offset_h: tuple[float, float, float],
        c_quat_h: tuple[float, float, float, float],
        position_offset_task: tuple[float, float, float],
        position_jitter_task: tuple[float, float, float],
        orientation_offset_rpy: tuple[float, float, float],
        orientation_jitter_rpy: tuple[float, float, float],
        hand_open_range: tuple[float, float],
        joint_seed_offsets: tuple[tuple[float, ...], ...],
        max_pose_attempts: int,
        max_iterations: int,
        damping: float,
        step_size: float,
        position_error_step: float,
        orientation_error_step: float,
        position_tolerance: float,
        orientation_tolerance: float,
        singular_value_min: float,
        joint_limit_margin: float,
        wrist_3_range: tuple[float, float],
    ) -> None:
        del (
            robot_cfg,
            arm_joint_names,
            hand_body_name,
            c_offset_h,
            c_quat_h,
            joint_seed_offsets,
        )
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        row_count = len(env_ids)
        if row_count == 0:
            return
        if max_pose_attempts <= 0 or max_iterations <= 0 or damping <= 0.0:
            raise ValueError("Stable reset attempts, iterations, and damping must be positive")
        if position_error_step <= 0.0 or orientation_error_step <= 0.0:
            raise ValueError("Stable reset Cartesian error steps must be positive")
        if not 0.0 < hand_open_range[0] <= hand_open_range[1] <= 1.0:
            raise ValueError("hand_open_range must be an increasing subset of (0, 1]")
        if any(value < 0.0 for value in position_jitter_task):
            raise ValueError("Position jitter magnitudes must be non-negative")
        if any(value < 0.0 for value in orientation_jitter_rpy):
            raise ValueError("Orientation jitter magnitudes must be non-negative")

        env.reset_ik_success[env_ids] = False
        env.reset_ik_attempts[env_ids] = 0
        env.reset_rejection_code[env_ids] = 0
        env.reset_rejection_counts[env_ids] = 0
        env.reset_valid_candidate_count[env_ids] = 0
        env.reset_selected_solution_cost[env_ids] = float("inf")
        env.reset_collision_free[env_ids] = False

        # A nearly open hand keeps the reset collision envelope predictable.
        # Both synergy coordinates are sampled independently inside one narrow
        # interval; the action term is synchronized to the resulting joints by
        # BlindSweepEnv._finalize_reset after the final kinematic refresh.
        env.initial_hand_synergy[env_ids].uniform_(*hand_open_range)
        hand_targets = _hand_targets_from_synergy(
            env.initial_hand_synergy[env_ids, 0],
            env.initial_hand_synergy[env_ids, 1],
            device=env.device,
        )

        success = torch.zeros(row_count, dtype=torch.bool, device=env.device)
        accepted_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        final_position_error = torch.full(
            (row_count,), torch.inf, dtype=torch.float32, device=env.device
        )
        final_orientation_error = torch.full_like(final_position_error, torch.inf)

        for pose_attempt in range(max_pose_attempts):
            pose_rows = torch.where(~success)[0]
            if pose_rows.numel() == 0:
                break
            pose_env_ids = env_ids[pose_rows]
            env.reset_episode_spec_attempts[pose_env_ids] = pose_attempt + 1
            desired_pos, desired_quat = self._sample_stable_pose(
                env,
                pose_env_ids,
                position_offset_task=position_offset_task,
                position_jitter_task=position_jitter_task,
                orientation_offset_rpy=orientation_offset_rpy,
                orientation_jitter_rpy=orientation_jitter_rpy,
            )
            env.desired_c_pos_w[pose_env_ids] = desired_pos
            env.desired_c_quat_w[pose_env_ids] = desired_quat

            workspace_ok = env.reset_workspace_mask(pose_env_ids)
            workspace_failed = pose_env_ids[~workspace_ok]
            env.reset_rejection_code[workspace_failed] = 6
            env.reset_rejection_counts[workspace_failed, 6] += 1
            eligible_rows = pose_rows[workspace_ok]

            for seed_offset in self.seed_offsets:
                seed_rows = eligible_rows[~success[eligible_rows]]
                if seed_rows.numel() == 0:
                    break
                seed_env_ids = env_ids[seed_rows]
                env.reset_ik_attempts[seed_env_ids] += 1

                seed_joint_pos = self.robot.data.default_joint_pos[seed_env_ids].clone()
                seed_joint_pos[:, self.arm_joint_ids] += seed_offset
                seed_joint_pos[:, self.hand_joint_ids] = hand_targets[seed_rows]
                arm_limits = self.robot.data.soft_joint_pos_limits[seed_env_ids][
                    :, self.arm_joint_ids
                ]
                seed_joint_pos[:, self.arm_joint_ids] = torch.clamp(
                    seed_joint_pos[:, self.arm_joint_ids],
                    arm_limits[..., 0] + joint_limit_margin,
                    arm_limits[..., 1] - joint_limit_margin,
                )
                self.robot.write_joint_state_to_sim(
                    seed_joint_pos,
                    torch.zeros_like(seed_joint_pos),
                    env_ids=seed_env_ids,
                )

                converged, pos_error, rot_error = self._solve_seed(
                    env,
                    seed_env_ids,
                    env.desired_c_pos_w[seed_env_ids],
                    env.desired_c_quat_w[seed_env_ids],
                    hand_targets[seed_rows],
                    max_iterations=max_iterations,
                    damping=damping,
                    step_size=step_size,
                    position_error_step=position_error_step,
                    orientation_error_step=orientation_error_step,
                    position_tolerance=position_tolerance,
                    orientation_tolerance=orientation_tolerance,
                    joint_limit_margin=joint_limit_margin,
                )
                final_position_error[seed_rows] = pos_error
                final_orientation_error[seed_rows] = rot_error
                nonconverged_ids = seed_env_ids[~converged]
                env.reset_rejection_code[nonconverged_ids] = 1
                env.reset_rejection_counts[nonconverged_ids, 1] += 1
                if not bool(converged.any()):
                    continue

                converged_rows = seed_rows[converged]
                converged_env_ids = env_ids[converged_rows]
                candidate_q = self.robot.data.joint_pos[converged_env_ids].clone()
                candidate_arm = candidate_q[:, self.arm_joint_ids]
                candidate_limits = self.robot.data.soft_joint_pos_limits[converged_env_ids][
                    :, self.arm_joint_ids
                ]
                inside_limits = torch.all(
                    (candidate_arm > candidate_limits[..., 0] + joint_limit_margin)
                    & (candidate_arm < candidate_limits[..., 1] - joint_limit_margin),
                    dim=-1,
                )
                wrist = candidate_arm[:, self.wrist_local_index]
                wrist_ok = (wrist >= wrist_3_range[0]) & (wrist <= wrist_3_range[1])
                c_jacobian = self._numerical_control_point_jacobian(converged_env_ids)
                nonsingular = (
                    torch.linalg.svdvals(c_jacobian)[:, -1] >= singular_value_min
                )
                clearance_ok = env.reset_clearance_mask(converged_env_ids)
                collision_free = torch.zeros_like(clearance_ok)
                if bool(clearance_ok.any()):
                    collision_free[clearance_ok] = env.reset_collision_free_mask(
                        converged_env_ids[clearance_ok]
                    )
                accepted = (
                    inside_limits
                    & wrist_ok
                    & nonsingular
                    & clearance_ok
                    & collision_free
                )

                rejected_collision = ~clearance_ok | ~collision_free
                collision_ids = converged_env_ids[rejected_collision]
                env.reset_rejection_code[collision_ids] = 5
                env.reset_rejection_counts[collision_ids, 5] += 1
                joint_ids = converged_env_ids[~inside_limits & ~rejected_collision]
                env.reset_rejection_code[joint_ids] = 2
                env.reset_rejection_counts[joint_ids, 2] += 1
                wrist_ids = converged_env_ids[
                    inside_limits & ~wrist_ok & ~rejected_collision
                ]
                env.reset_rejection_code[wrist_ids] = 3
                env.reset_rejection_counts[wrist_ids, 3] += 1
                singular_ids = converged_env_ids[
                    ~nonsingular & inside_limits & wrist_ok & ~rejected_collision
                ]
                env.reset_rejection_code[singular_ids] = 4
                env.reset_rejection_counts[singular_ids, 4] += 1

                if bool(accepted.any()):
                    accepted_rows = converged_rows[accepted]
                    accepted_ids = converged_env_ids[accepted]
                    accepted_joint_pos[accepted_rows] = candidate_q[accepted]
                    success[accepted_rows] = True
                    env.reset_ik_success[accepted_ids] = True
                    env.reset_collision_free[accepted_ids] = True
                    env.reset_valid_candidate_count[accepted_ids] = 1
                    env.reset_rejection_code[accepted_ids] = 0
                    delta = candidate_arm[accepted] - self.robot.data.default_joint_pos[
                        accepted_ids
                    ][:, self.arm_joint_ids]
                    wrapped_delta = torch.atan2(torch.sin(delta), torch.cos(delta))
                    env.reset_selected_solution_cost[accepted_ids] = torch.sum(
                        wrapped_delta.square(), dim=-1
                    )

        if not bool(success.all()):
            failed_rows = torch.where(~success)[0]
            body_names = self.robot.body_names
            details = [
                {
                    "env_id": int(env_ids[row]),
                    "desired_c_pos_w": env.desired_c_pos_w[env_ids[row]].tolist(),
                    "position_error": float(final_position_error[row]),
                    "orientation_error": float(final_orientation_error[row]),
                    "minimum_clearance": float(env.reset_min_clearance_m[env_ids[row]]),
                    "minimum_board_clearance": float(
                        env.reset_min_board_clearance_m[env_ids[row]]
                    ),
                    "minimum_object_clearance": float(
                        env.reset_min_object_clearance_m[env_ids[row]]
                    ),
                    "surface_mode": int(env.surface_mode[env_ids[row]]),
                    "command_direction": env.command_direction_w[env_ids[row]].tolist(),
                    "object_position_s": (
                        env.object_initial_pos_w[env_ids[row]]
                        - env.scene.env_origins[env_ids[row]]
                    ).tolist(),
                    "cube_overlap_bodies": [
                        body_names[index]
                        for index in torch.where(
                            env._reset_last_cube_overlap[env_ids[row]]
                        )[0].tolist()
                    ],
                    "board_overlap_bodies": [
                        body_names[index]
                        for index in torch.where(
                            env._reset_last_board_overlap[env_ids[row]]
                        )[0].tolist()
                    ],
                    "rejection_code": int(env.reset_rejection_code[env_ids[row]]),
                    "attempts": int(env.reset_ik_attempts[env_ids[row]]),
                }
                for row in failed_rows[:8].tolist()
            ]
            raise RuntimeError(
                f"Stable offset reset failed for {len(failed_rows)}/{row_count} environments "
                f"after {max_pose_attempts} pose attempts. First failures: {details}"
            )

        accepted_joint_pos[:, self.hand_joint_ids] = hand_targets
        zero_velocity = torch.zeros_like(accepted_joint_pos)
        self.robot.write_joint_state_to_sim(
            accepted_joint_pos, zero_velocity, env_ids=env_ids
        )
        self.robot.set_joint_position_target(accepted_joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(zero_velocity, env_ids=env_ids)
        self.robot.set_joint_effort_target(
            torch.zeros((row_count, len(self.arm_joint_ids)), device=env.device),
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )

    def _sample_stable_pose(
        self,
        env: "BlindSweepEnv",
        env_ids: torch.Tensor,
        *,
        position_offset_task: tuple[float, float, float],
        position_jitter_task: tuple[float, float, float],
        orientation_offset_rpy: tuple[float, float, float],
        orientation_jitter_rpy: tuple[float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a small collision-averse pose region behind the Cube."""

        count = len(env_ids)
        direction = env.command_direction_w[env_ids]
        tangent = torch.stack(
            (-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])),
            dim=-1,
        )
        up = torch.zeros_like(direction)
        up[:, 2] = 1.0
        base_offset = torch.as_tensor(
            position_offset_task, dtype=torch.float32, device=env.device
        )
        jitter_limit = torch.as_tensor(
            position_jitter_task, dtype=torch.float32, device=env.device
        )
        task_offset = base_offset + (2.0 * torch.rand((count, 3), device=env.device) - 1.0) * jitter_limit
        # C is shifted from the hand center along Hand +Y. Reversing Hand +Y
        # for a dorsal episode would otherwise move the same physical hand
        # center 2*|C_y| closer to the Cube. Compensate the C target so palm
        # and dorsal starts have the same collision-safe hand standoff.
        dorsal = (env.surface_mode[env_ids] == 1).to(dtype=task_offset.dtype)
        task_offset[:, 0] += dorsal * (2.0 * torch.abs(self.c_offset_h[1]))
        target_pos = env.scene["target_object"].data.root_pos_w[env_ids]
        desired_pos = (
            target_pos
            - task_offset[:, 0:1] * direction
            + task_offset[:, 1:2] * tangent
            + task_offset[:, 2:3] * up
        )

        # H +Y is the palmar outward normal; dorsal episodes reverse H +Y.
        mode_sign = torch.where(
            env.surface_mode[env_ids] == 0,
            torch.ones(count, device=env.device),
            -torch.ones(count, device=env.device),
        )
        hand_y = mode_sign.unsqueeze(-1) * direction
        # Choose the roll-equivalent orientation whose +Z finger axis always
        # points into the shelf (-world X). Since C is offset along Hand +Z,
        # this keeps the wrist on the robot side of C for both sweep
        # directions instead of demanding a near-fully-extended arm in half
        # of the episodes. Hand +Y, and therefore the requested palm/dorsal
        # contact normal, is unchanged.
        hand_x = torch.sign(hand_y[:, 1:2]) * up
        hand_z = torch.nn.functional.normalize(
            torch.linalg.cross(hand_x, hand_y, dim=-1), dim=-1
        )
        hand_y = torch.linalg.cross(hand_z, hand_x, dim=-1)
        nominal_hand_quat = math_utils.quat_unique(
            math_utils.quat_from_matrix(torch.stack((hand_x, hand_y, hand_z), dim=-1))
        )
        base_rpy = torch.as_tensor(
            orientation_offset_rpy, dtype=torch.float32, device=env.device
        )
        rpy_limit = torch.as_tensor(
            orientation_jitter_rpy, dtype=torch.float32, device=env.device
        )
        rpy = base_rpy + (2.0 * torch.rand((count, 3), device=env.device) - 1.0) * rpy_limit
        hand_quat = math_utils.quat_unique(
            math_utils.quat_mul(
                nominal_hand_quat,
                math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2]),
            )
        )
        desired_c_quat = math_utils.quat_unique(
            math_utils.quat_mul(hand_quat, self.c_quat_h.expand(count, -1))
        )
        return desired_pos, desired_c_quat

    def _solve_seed(
        self,
        env: "BlindSweepEnv",
        env_ids: torch.Tensor,
        desired_pos: torch.Tensor,
        desired_quat: torch.Tensor,
        hand_targets: torch.Tensor,
        *,
        max_iterations: int,
        damping: float,
        step_size: float,
        position_error_step: float,
        orientation_error_step: float,
        position_tolerance: float,
        orientation_tolerance: float,
        joint_limit_margin: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve one seed using kinematic refreshes, never a physics forward."""

        count = len(env_ids)
        converged = torch.zeros(count, dtype=torch.bool, device=env.device)
        identity = torch.eye(6, device=env.device).unsqueeze(0)

        for _ in range(max_iterations):
            active_rows = torch.where(~converged)[0]
            if active_rows.numel() == 0:
                break
            active_ids = env_ids[active_rows]
            hand_pos = self.robot.data.body_pos_w[active_ids, self.hand_body_id]
            hand_quat = self.robot.data.body_quat_w[active_ids, self.hand_body_id]
            c_pos, c_quat = math_utils.combine_frame_transforms(
                hand_pos,
                hand_quat,
                self.c_offset_h.expand(len(active_ids), -1),
                self.c_quat_h.expand(len(active_ids), -1),
            )
            pos_error, rot_error = math_utils.compute_pose_error(
                c_pos,
                c_quat,
                desired_pos[active_rows],
                desired_quat[active_rows],
                rot_error_type="axis_angle",
            )
            reached = (
                torch.linalg.vector_norm(pos_error, dim=-1) <= position_tolerance
            ) & (
                torch.linalg.vector_norm(rot_error, dim=-1) <= orientation_tolerance
            )
            converged[active_rows[reached]] = True
            solve_rows = active_rows[~reached]
            if solve_rows.numel() == 0:
                break
            solve_ids = env_ids[solve_rows]
            solve_pos_error = pos_error[~reached]
            solve_rot_error = rot_error[~reached]
            # Pose errors close to pi make a one-shot axis-angle DLS update
            # jump between equivalent quaternion branches and immediately
            # saturate several joints. Follow the exact final pose through
            # short Cartesian steps instead; final tolerances stay unchanged.
            pos_norm = torch.linalg.vector_norm(solve_pos_error, dim=-1, keepdim=True)
            rot_norm = torch.linalg.vector_norm(solve_rot_error, dim=-1, keepdim=True)
            solve_pos_error = solve_pos_error * torch.clamp(
                position_error_step / pos_norm.clamp_min(1.0e-9), max=1.0
            )
            solve_rot_error = solve_rot_error * torch.clamp(
                orientation_error_step / rot_norm.clamp_min(1.0e-9), max=1.0
            )
            solve_error = torch.cat((solve_pos_error, solve_rot_error), dim=-1)
            c_jacobian = self._numerical_control_point_jacobian(solve_ids)
            dls = torch.linalg.solve(
                c_jacobian @ c_jacobian.transpose(1, 2) + damping**2 * identity,
                solve_error.unsqueeze(-1),
            )
            delta_q = (c_jacobian.transpose(1, 2) @ dls).squeeze(-1)
            delta_q = torch.clamp(step_size * delta_q, min=-0.18, max=0.18)
            next_q = self.robot.data.joint_pos[solve_ids].clone()
            next_q[:, self.arm_joint_ids] += delta_q
            next_q[:, self.hand_joint_ids] = hand_targets[solve_rows]
            limits = self.robot.data.soft_joint_pos_limits[solve_ids][:, self.arm_joint_ids]
            next_q[:, self.arm_joint_ids] = torch.clamp(
                next_q[:, self.arm_joint_ids],
                limits[..., 0] + joint_limit_margin,
                limits[..., 1] - joint_limit_margin,
            )
            self.robot.write_joint_state_to_sim(
                next_q, torch.zeros_like(next_q), env_ids=solve_ids
            )

        hand_pos = self.robot.data.body_pos_w[env_ids, self.hand_body_id]
        hand_quat = self.robot.data.body_quat_w[env_ids, self.hand_body_id]
        c_pos, c_quat = math_utils.combine_frame_transforms(
            hand_pos,
            hand_quat,
            self.c_offset_h.expand(count, -1),
            self.c_quat_h.expand(count, -1),
        )
        pos_error, rot_error = math_utils.compute_pose_error(
            c_pos,
            c_quat,
            desired_pos,
            desired_quat,
            rot_error_type="axis_angle",
        )
        pos_norm = torch.linalg.vector_norm(pos_error, dim=-1)
        rot_norm = torch.linalg.vector_norm(rot_error, dim=-1)
        converged = (pos_norm <= position_tolerance) & (rot_norm <= orientation_tolerance)
        return converged, pos_norm, rot_norm

    def _numerical_control_point_jacobian(
        self, env_ids: torch.Tensor, epsilon: float = 1.0e-3
    ) -> torch.Tensor:
        """Evaluate C's geometric Jacobian after tensor joint teleports.

        PhysX Jacobian buffers are refreshed by a simulation forward, whereas
        link transforms support a cheaper articulation-kinematic refresh after
        ``write_joint_position_to_sim``.  A six-column forward difference uses
        those live transforms and then restores the exact candidate state.  It
        is reset-only, batched over all selected environments, and does not
        advance physics time.
        """

        if epsilon <= 0.0:
            raise ValueError("Finite-difference epsilon must be positive")
        count = len(env_ids)
        base_q = self.robot.data.joint_pos[env_ids].clone()
        hand_pos = self.robot.data.body_pos_w[env_ids, self.hand_body_id]
        hand_quat = self.robot.data.body_quat_w[env_ids, self.hand_body_id]
        base_pos, base_quat = math_utils.combine_frame_transforms(
            hand_pos,
            hand_quat,
            self.c_offset_h.expand(count, -1),
            self.c_quat_h.expand(count, -1),
        )
        columns: list[torch.Tensor] = []
        for joint_id in self.arm_joint_ids:
            perturbed_q = base_q.clone()
            perturbed_q[:, joint_id] += epsilon
            self.robot.write_joint_position_to_sim(perturbed_q, env_ids=env_ids)
            perturbed_hand_pos = self.robot.data.body_pos_w[env_ids, self.hand_body_id]
            perturbed_hand_quat = self.robot.data.body_quat_w[env_ids, self.hand_body_id]
            perturbed_pos, perturbed_quat = math_utils.combine_frame_transforms(
                perturbed_hand_pos,
                perturbed_hand_quat,
                self.c_offset_h.expand(count, -1),
                self.c_quat_h.expand(count, -1),
            )
            _, angular_delta = math_utils.compute_pose_error(
                base_pos,
                base_quat,
                base_pos,
                perturbed_quat,
                rot_error_type="axis_angle",
            )
            columns.append(
                torch.cat(
                    (
                        (perturbed_pos - base_pos) / epsilon,
                        angular_delta / epsilon,
                    ),
                    dim=-1,
                )
            )
        self.robot.write_joint_position_to_sim(base_q, env_ids=env_ids)
        return torch.stack(columns, dim=-1)


class ConditionalPoseIKReset(ManagerTermBase):
    """Manipulator reset using full-pose, six-joint damped least-squares IK.

    The selected palm/dorsal face is encoded in the desired hand orientation
    sampled by the environment.  Wrist-3 is never modified by adding an
    unconditional pi rotation.  Multiple bounded seeds are evaluated and the
    solution is rejected on pose error, wrist range, joint-limit margin, or a
    small Jacobian singular value.

    Online rejection uses conservative robot--object/board geometry from the
    current FK state without a hidden physics step.  Accepted poses are fully
    certified before the episode is exposed to the policy.
    """

    def __init__(self, cfg: EventTermCfg, env: "BlindSweepEnv"):
        super().__init__(cfg=cfg, env=env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["robot_cfg"].name]
        self.target: RigidObject = env.scene[params["object_cfg"].name]
        self.arm_joint_ids, self.arm_joint_names = self.robot.find_joints(
            list(params["arm_joint_names"]), preserve_order=True
        )
        if tuple(self.arm_joint_names) != tuple(params["arm_joint_names"]):
            raise RuntimeError(f"Unexpected arm joint order: {self.arm_joint_names}")
        self.hand_joint_ids, self.hand_joint_names = self.robot.find_joints(
            list(HAND_JOINT_NAMES), preserve_order=True
        )
        if tuple(self.hand_joint_names) != HAND_JOINT_NAMES:
            raise RuntimeError(f"Unexpected Inspire joint order: {self.hand_joint_names}")
        body_ids, body_names = self.robot.find_bodies(params["hand_body_name"], preserve_order=True)
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one hand body, got {body_names}")
        self.hand_body_id = int(body_ids[0])
        self.jacobian_body_id = self.hand_body_id - 1 if self.robot.is_fixed_base else self.hand_body_id
        self.jacobian_joint_ids = (
            self.arm_joint_ids if self.robot.is_fixed_base else [joint_id + 6 for joint_id in self.arm_joint_ids]
        )
        self.wrist_local_index = self.arm_joint_names.index("wrist_3_joint")
        self.c_offset_h = torch.tensor(params["c_offset_h"], dtype=torch.float32, device=env.device)
        self.c_quat_h = torch.tensor(params["c_quat_h"], dtype=torch.float32, device=env.device)
        self.seed_offsets = torch.tensor(params["joint_seed_offsets"], dtype=torch.float32, device=env.device)
        if self.seed_offsets.ndim != 2 or self.seed_offsets.shape[1] != 6:
            raise ValueError("Every IK seed offset must contain six arm values.")

    def __call__(
        self,
        env: "BlindSweepEnv",
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        arm_joint_names: tuple[str, ...],
        hand_body_name: str,
        c_offset_h: tuple[float, float, float],
        c_quat_h: tuple[float, float, float, float],
        joint_seed_offsets: tuple[tuple[float, ...], ...],
        max_iterations: int,
        damping: float,
        step_size: float,
        position_tolerance: float,
        orientation_tolerance: float,
        singular_value_min: float,
        joint_limit_margin: float,
        wrist_3_range: tuple[float, float],
        wrist_3_cost_weight: float,
        max_spec_resamples: int,
    ) -> None:
        del robot_cfg, object_cfg, arm_joint_names, hand_body_name, c_offset_h, c_quat_h, joint_seed_offsets
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if damping <= 0.0 or max_iterations <= 0 or max_spec_resamples <= 0:
            raise ValueError("IK damping, iterations and specification retries must be positive.")
        if wrist_3_cost_weight < 0.0:
            raise ValueError("wrist_3_cost_weight must be non-negative.")

        pending = env_ids.clone()
        env.reset_ik_success[env_ids] = False
        env.reset_ik_attempts[env_ids] = 0
        env.reset_rejection_code[env_ids] = 0
        env.reset_rejection_counts[env_ids] = 0
        env.reset_valid_candidate_count[env_ids] = 0
        env.reset_selected_solution_cost[env_ids] = float("inf")
        env.reset_collision_free[env_ids] = False

        for spec_attempt in range(max_spec_resamples):
            if pending.numel() == 0:
                break
            if spec_attempt > 0:
                env.sample_episode_specs(pending)
                reset_episode_scene(env, pending)
            env.reset_episode_spec_attempts[pending] = spec_attempt + 1
            # Hand shape is part of the collision geometry of the reset
            # candidate, so write the sampled posture before evaluating IK.
            self._write_hand_state(env, pending)
            workspace_ok = env.reset_workspace_mask(pending)
            workspace_failed_ids = pending[~workspace_ok]
            env.reset_rejection_code[workspace_failed_ids] = 6
            env.reset_rejection_counts[workspace_failed_ids, 6] += 1
            success = torch.zeros(len(pending), dtype=torch.bool, device=env.device)
            if workspace_ok.any():
                success[workspace_ok] = self._solve_spec(
                    env,
                    pending[workspace_ok],
                    max_iterations=max_iterations,
                    damping=damping,
                    step_size=step_size,
                    position_tolerance=position_tolerance,
                    orientation_tolerance=orientation_tolerance,
                    singular_value_min=singular_value_min,
                    joint_limit_margin=joint_limit_margin,
                    wrist_3_range=wrist_3_range,
                    wrist_3_cost_weight=wrist_3_cost_weight,
                )
            env.reset_ik_success[pending] = success
            pending = pending[~success]

        if pending.numel() > 0:
            details = [
                {
                    "env_id": int(index),
                    "mode": int(env.surface_mode[index]),
                    "theta": float(env.command_angle[index]),
                    "ik_attempts": int(env.reset_ik_attempts[index]),
                    "rejection_code": int(env.reset_rejection_code[index]),
                    "rejection_counts": env.reset_rejection_counts[index].tolist(),
                    "proxy_min_clearance_m": float(env.reset_min_clearance_m[index]),
                    "last_cube_obb_overlap": [
                        env.scene["robot"].body_names[body_id]
                        for body_id in env._reset_last_cube_overlap[index].nonzero(
                            as_tuple=False
                        ).squeeze(-1).tolist()
                    ],
                    "last_board_obb_overlap": [
                        env.scene["robot"].body_names[body_id]
                        for body_id in env._reset_last_board_overlap[index].nonzero(
                            as_tuple=False
                        ).squeeze(-1).tolist()
                    ],
                }
                for index in pending[:8].tolist()
            ]
            raise RuntimeError(
                f"Blind-sweep reset failed for {len(pending)}/{len(env_ids)} environments after "
                f"{max_spec_resamples} episode-spec samples. First failures: {details}"
            )

        # Initialize the hand with bounded, non-contact synergy values and make
        # every drive target equal to the state that was actually written.
        self._write_hand_state(env, env_ids)
        self.robot.set_joint_effort_target(
            torch.zeros((len(env_ids), len(self.arm_joint_ids)), device=env.device),
            joint_ids=self.arm_joint_ids,
            env_ids=env_ids,
        )

    def _write_hand_state(self, env: "BlindSweepEnv", env_ids: torch.Tensor) -> None:
        """Apply the sampled 2-D hand posture to state and drive targets."""

        hand_targets = _hand_targets_from_synergy(
            env.initial_hand_synergy[env_ids, 0],
            env.initial_hand_synergy[env_ids, 1],
            device=env.device,
        )
        joint_pos = self.robot.data.joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.hand_joint_ids] = hand_targets
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(
            hand_targets, joint_ids=self.hand_joint_ids, env_ids=env_ids
        )
        self.robot.set_joint_velocity_target(
            torch.zeros_like(hand_targets), joint_ids=self.hand_joint_ids, env_ids=env_ids
        )

    def _solve_spec(
        self,
        env: "BlindSweepEnv",
        env_ids: torch.Tensor,
        *,
        max_iterations: int,
        damping: float,
        step_size: float,
        position_tolerance: float,
        orientation_tolerance: float,
        singular_value_min: float,
        joint_limit_margin: float,
        wrist_3_range: tuple[float, float],
        wrist_3_cost_weight: float,
    ) -> torch.Tensor:
        row_count = len(env_ids)
        identity = torch.eye(6, device=env.device).unsqueeze(0)
        best_score = torch.full((row_count,), float("inf"), device=env.device)
        best_q_arm = torch.zeros((row_count, 6), device=env.device)

        # Dorsal episodes begin from a different natural shoulder/elbow seed,
        # not a hard-coded pi change at wrist-3.
        mode_seed = torch.zeros((row_count, 6), device=env.device)
        dorsal = env.surface_mode[env_ids].bool()
        mode_seed[dorsal, 0] = -0.45
        mode_seed[dorsal, 1] = -0.20
        mode_seed[dorsal, 2] = 0.35
        reference_arm = self.robot.data.default_joint_pos[env_ids][:, self.arm_joint_ids] + mode_seed
        sampled_hand_targets = _hand_targets_from_synergy(
            env.initial_hand_synergy[env_ids, 0],
            env.initial_hand_synergy[env_ids, 1],
            device=env.device,
        )
        rows = torch.arange(row_count, device=env.device)

        # Evaluate every seed.  Valid solutions are compared in wrapped joint
        # distance, with an explicit additional wrist-3 cost, instead of
        # accepting whichever numerical seed happens to converge first.
        for seed_offset in self.seed_offsets:
            seed_rejected = torch.zeros(row_count, dtype=torch.bool, device=env.device)
            env.reset_ik_attempts[env_ids] += 1
            q = self.robot.data.default_joint_pos[env_ids].clone()
            q[:, self.arm_joint_ids] += mode_seed + seed_offset
            # The sampled hand posture is part of the candidate collision
            # geometry.  Do not let the arm seed's default full-joint state
            # silently replace it with the fully-open hand.
            q[:, self.hand_joint_ids] = sampled_hand_targets
            limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_ids]
            q[:, self.arm_joint_ids] = torch.clamp(
                q[:, self.arm_joint_ids],
                limits[..., 0] + joint_limit_margin,
                limits[..., 1] - joint_limit_margin,
            )
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

            for _ in range(max_iterations):
                env.sim.forward()
                self.robot.update(0.0)
                active_rows = rows[~seed_rejected]
                if active_rows.numel() == 0:
                    break
                active_env_ids = env_ids[active_rows]
                hand_pos = self.robot.data.body_pos_w[active_env_ids, self.hand_body_id]
                hand_quat = self.robot.data.body_quat_w[active_env_ids, self.hand_body_id]
                c_offset = self.c_offset_h.expand(len(active_env_ids), -1)
                c_quat_offset = self.c_quat_h.expand(len(active_env_ids), -1)
                c_pos, c_quat = math_utils.combine_frame_transforms(
                    hand_pos, hand_quat, c_offset, c_quat_offset
                )
                pos_error, rot_error = math_utils.compute_pose_error(
                    c_pos,
                    c_quat,
                    env.desired_c_pos_w[active_env_ids],
                    env.desired_c_quat_w[active_env_ids],
                    rot_error_type="axis_angle",
                )
                converged = (torch.linalg.vector_norm(pos_error, dim=-1) <= position_tolerance) & (
                    torch.linalg.vector_norm(rot_error, dim=-1) <= orientation_tolerance
                )

                # Singularity is an acceptance criterion too: computing it
                # only for non-converged rows would admit a converged but
                # uncontrollable pose.
                jacobian = self.robot.root_physx_view.get_jacobians()[
                    active_env_ids, self.jacobian_body_id, :, :
                ][..., self.jacobian_joint_ids]
                r_hc_w = math_utils.quat_apply(
                    hand_quat, self.c_offset_h.expand(len(active_env_ids), -1)
                )
                # PhysX evaluates the link's linear Jacobian at its COM, not
                # at H's actor origin.  Shift directly from that COM to C so
                # both the DLS update and singular-value rejection use the
                # same physical point as the pose error.
                r_hcom_w = math_utils.quat_apply(
                    hand_quat,
                    self.robot.data.body_com_pos_b[
                        active_env_ids, self.hand_body_id
                    ],
                )
                r_comc_w = r_hc_w - r_hcom_w
                linear_offset = torch.linalg.cross(
                    jacobian[:, 3:, :].transpose(1, 2),
                    r_comc_w.unsqueeze(1),
                    dim=-1,
                ).transpose(1, 2)
                c_jacobian = torch.cat(
                    (jacobian[:, :3, :] + linear_offset, jacobian[:, 3:, :]), dim=1
                )
                singular = torch.linalg.svdvals(c_jacobian)[:, -1] < singular_value_min
                singular_env_ids = active_env_ids[singular]
                env.reset_rejection_code[singular_env_ids] = 4
                env.reset_rejection_counts[singular_env_ids, 4] += 1
                seed_rejected[active_rows[singular]] = True

                if converged.any():
                    converged_rows = active_rows[converged]
                    converged_env_ids = active_env_ids[converged]
                    candidate_q = self.robot.data.joint_pos[converged_env_ids][:, self.arm_joint_ids]
                    candidate_limits = self.robot.data.soft_joint_pos_limits[converged_env_ids][
                        :, self.arm_joint_ids
                    ]
                    inside_limits = torch.all(
                        (candidate_q > candidate_limits[..., 0] + joint_limit_margin)
                        & (candidate_q < candidate_limits[..., 1] - joint_limit_margin),
                        dim=-1,
                    )
                    wrist = candidate_q[:, self.wrist_local_index]
                    wrist_ok = (wrist >= wrist_3_range[0]) & (wrist <= wrist_3_range[1])
                    clearance_ok = env.reset_clearance_mask(converged_env_ids)
                    collision_free = torch.zeros_like(clearance_ok)
                    if clearance_ok.any():
                        collision_free[clearance_ok] = env.reset_collision_free_mask(
                            converged_env_ids[clearance_ok]
                        )
                    converged_singular = singular[converged]
                    accepted = (
                        inside_limits
                        & wrist_ok
                        & clearance_ok
                        & collision_free
                        & ~converged_singular
                    )

                    if accepted.any():
                        accepted_rows = converged_rows[accepted]
                        accepted_env_ids = converged_env_ids[accepted]
                        accepted_q = candidate_q[accepted]
                        delta = accepted_q - reference_arm[accepted_rows]
                        wrapped_delta = torch.atan2(torch.sin(delta), torch.cos(delta))
                        score = torch.sum(wrapped_delta.square(), dim=-1)
                        score += wrist_3_cost_weight * wrapped_delta[:, self.wrist_local_index].square()
                        env.reset_valid_candidate_count[accepted_env_ids] += 1
                        improved = score < best_score[accepted_rows]
                        improved_rows = accepted_rows[improved]
                        best_score[improved_rows] = score[improved]
                        best_q_arm[improved_rows] = accepted_q[improved]

                    clearance_failed = (
                        (~clearance_ok | ~collision_free) & ~converged_singular
                    )
                    clearance_failed_ids = converged_env_ids[clearance_failed]
                    env.reset_rejection_code[clearance_failed_ids] = 5
                    env.reset_rejection_counts[clearance_failed_ids, 5] += 1

                    joint_failed = (
                        ~inside_limits
                        & ~converged_singular
                        & clearance_ok
                        & collision_free
                    )
                    joint_failed_ids = converged_env_ids[joint_failed]
                    env.reset_rejection_code[joint_failed_ids] = 2
                    env.reset_rejection_counts[joint_failed_ids, 2] += 1

                    wrist_failed = (
                        inside_limits
                        & ~wrist_ok
                        & ~converged_singular
                        & clearance_ok
                        & collision_free
                    )
                    wrist_failed_ids = converged_env_ids[wrist_failed]
                    env.reset_rejection_code[wrist_failed_ids] = 3
                    env.reset_rejection_counts[wrist_failed_ids, 3] += 1

                    # Every converged candidate has now been either stored or
                    # assigned a rejection reason for this seed.
                    seed_rejected[converged_rows] = True

                iterating = ~converged & ~singular
                if not iterating.any():
                    break
                safe_env_ids = active_env_ids[iterating]
                safe_error = torch.cat((pos_error[iterating], rot_error[iterating]), dim=-1)
                safe_jacobian = c_jacobian[iterating]
                dls = torch.linalg.solve(
                    safe_jacobian @ safe_jacobian.transpose(1, 2) + damping**2 * identity,
                    safe_error.unsqueeze(-1),
                )
                delta_q = (safe_jacobian.transpose(1, 2) @ dls).squeeze(-1)
                delta_q = torch.clamp(step_size * delta_q, min=-0.18, max=0.18)
                next_q = self.robot.data.joint_pos[safe_env_ids].clone()
                next_q[:, self.arm_joint_ids] += delta_q
                next_limits = self.robot.data.soft_joint_pos_limits[safe_env_ids][:, self.arm_joint_ids]
                next_q[:, self.arm_joint_ids] = torch.clamp(
                    next_q[:, self.arm_joint_ids],
                    next_limits[..., 0] + joint_limit_margin,
                    next_limits[..., 1] - joint_limit_margin,
                )
                self.robot.write_joint_state_to_sim(
                    next_q, torch.zeros_like(next_q), env_ids=safe_env_ids
                )

            nonconverged_rows = rows[~seed_rejected]
            nonconverged_ids = env_ids[nonconverged_rows]
            env.reset_rejection_code[nonconverged_ids] = 1
            env.reset_rejection_counts[nonconverged_ids, 1] += 1

        success = torch.isfinite(best_score)
        final_q = self.robot.data.joint_pos[env_ids].clone()
        final_arm = final_q[:, self.arm_joint_ids]
        final_arm[success] = best_q_arm[success]
        final_q[:, self.arm_joint_ids] = final_arm
        final_q[:, self.hand_joint_ids] = sampled_hand_targets
        self.robot.write_joint_state_to_sim(final_q, torch.zeros_like(final_q), env_ids=env_ids)
        self.robot.set_joint_position_target(final_q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(final_q), env_ids=env_ids)
        env.reset_selected_solution_cost[env_ids[success]] = best_score[success]
        env.reset_rejection_code[env_ids[success]] = 0
        env.reset_collision_free[env_ids] = success
        return success


__all__ = ["ConditionalPoseIKReset", "StableOffsetPoseReset", "reset_episode_scene"]
