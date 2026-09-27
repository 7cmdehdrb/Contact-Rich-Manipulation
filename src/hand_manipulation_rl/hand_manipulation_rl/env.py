"""Manager-based RL environment with reset ordering and substep latching.

The stock ManagerBasedRLEnv resets commands after reset events.  This task must
construct the IK pose from the exact episode command, so this subclass samples
one authoritative episode specification *before* the two reset events, keeps it
outside CommandManager, and finalizes C0 only after every manager has reset.
It also observes hard safety thresholds at every physics substep.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .assets.robot import (
    FT_MEASUREMENT_CHILD_BODY_NAME,
    FT_SENSOR_BODY_NAME,
    HAND_BASE_BODY_NAME,
    HAND_CONTACT_BODY_NAMES,
    PALM_SENSOR_BODY_NAMES,
    ROBOT_CONTACT_BODY_NAMES,
)
from .mdp.commands import direction_from_angle, sample_sweep_angles
from .mdp.rewards import upright_tilt_angle
from .mdp.terminations import TerminationReason, object_out_of_bounds, success_from_positions
from .math_utils import oriented_box_overlap
from .sensors import DORSAL_PARENT_BODY_NAMES
from .sensors import (
    BilateralTactileReader,
    FixedJointWrenchReader,
    hand_object_resultant_normal,
    robot_board_contact_forces,
)

if TYPE_CHECKING:
    from .env_cfg import BlindSweepEnvCfg


class BlindSweepEnv(ManagerBasedRLEnv):
    """Blind single-cube sweeping with bilateral tactile and wrist wrench."""

    cfg: "BlindSweepEnvCfg"

    def __init__(self, cfg: "BlindSweepEnvCfg", render_mode: str | None = None, **kwargs):
        del kwargs
        count = cfg.scene.num_envs
        device = cfg.sim.device
        zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
        false = lambda: torch.zeros(count, dtype=torch.bool, device=device)

        # Fixed episode specification and reset targets.
        self.surface_mode = torch.zeros(count, dtype=torch.long, device=device)
        self.command_angle = zeros(count)
        self.command_distance = zeros(count)
        self.command_direction_w = zeros(count, 3)
        self.object_initial_pos_w = zeros(count, 3)
        self.object_initial_quat_w = zeros(count, 4)
        self.object_initial_quat_w[:, 0] = 1.0
        self.goal_pos_w = zeros(count, 3)
        self.desired_c_pos_w = zeros(count, 3)
        self.desired_c_quat_w = zeros(count, 4)
        self.desired_c_quat_w[:, 0] = 1.0
        self.initial_hand_synergy = zeros(count, 2)

        # Actual references captured only after FK has consumed the reset state.
        self.c0_pos_w = zeros(count, 3)
        self.c0_quat_w = zeros(count, 4)
        self.c0_quat_w[:, 0] = 1.0
        self.initial_object_pos_c0 = zeros(count, 3)
        self.initial_object_upright_w = zeros(count, 3)
        self.initial_object_upright_w[:, 2] = 1.0
        self.initial_object_height_w = zeros(count)
        self.initial_arm_joint_pos = zeros(count, 6)
        self.initial_hand_actual_synergy = zeros(count, 2)

        # Policy/action/sensor history.
        self.last_policy_action = zeros(count, 8)
        self.current_policy_action = zeros(count, 8)
        self.previous_policy_action = zeros(count, 8)
        self.has_previous_policy_action = false()
        self.sensor_valid = false()
        # Reset observations use a deliberate, collision-certified zero
        # sensor buffer.  This flag distinguishes that initialized value from
        # the first live contact/wrench tensors produced by physics.
        self.sensor_data_fresh = false()
        self.invalid_reset_latched = false()

        # Physics-substep diagnostics and hard-threshold latches.
        self.board_force = zeros(count)
        self.board_force_peak = zeros(count)
        self.board_link_force_peak = zeros(count)
        self.board_link_index_peak = torch.zeros(count, dtype=torch.long, device=device)
        self.board_link_position_peak_w = zeros(count, 3)
        self.board_contact_c_position_peak_w = zeros(count, 3)
        self.current_tilt = zeros(count)
        self.tilt_peak = zeros(count)
        self.current_height_increase = zeros(count)
        self.height_peak = zeros(count)
        self.board_hard_latched = false()
        self.tilt_hard_latched = false()
        self.height_hard_latched = false()
        self.invalid_sim_latched = false()
        self.failure_reason = torch.zeros(count, dtype=torch.long, device=device)
        self.last_termination_reason = torch.zeros(count, dtype=torch.long, device=device)
        self.episode_physics_substep_count = torch.zeros(count, dtype=torch.long, device=device)
        self.first_selected_contact_substep = torch.full(
            (count,), -1, dtype=torch.long, device=device
        )
        self.selected_contact_previous = false()
        self.selected_contact_loss_count = torch.zeros(count, dtype=torch.long, device=device)
        self.arm_torque_saturation_steps = torch.zeros(count, dtype=torch.long, device=device)
        self.hand_action_saturation_steps = torch.zeros(count, dtype=torch.long, device=device)

        # Reset audit information (never included in actor observations).
        self.reset_ik_success = false()
        self.reset_ik_attempts = torch.zeros(count, dtype=torch.long, device=device)
        self.reset_rejection_code = torch.zeros(count, dtype=torch.long, device=device)
        self.reset_rejection_counts = torch.zeros(count, 7, dtype=torch.long, device=device)
        self.reset_valid_candidate_count = torch.zeros(count, dtype=torch.long, device=device)
        self.reset_selected_solution_cost = torch.full(
            (count,), float("inf"), dtype=torch.float32, device=device
        )
        self.reset_path_min_reach_m = zeros(count)
        self.reset_path_max_reach_m = zeros(count)
        self.reset_episode_spec_attempts = torch.zeros(
            count, dtype=torch.long, device=device
        )
        self.reset_spec_attempts = torch.zeros(count, dtype=torch.long, device=device)
        self.reset_position_error = zeros(count)
        self.reset_orientation_error = zeros(count)
        self.reset_min_clearance_m = zeros(count)
        self.reset_min_board_clearance_m = zeros(count)
        self.reset_min_object_clearance_m = zeros(count)
        self.reset_collision_free = false()
        self._tactile_reader: BilateralTactileReader | None = None
        self._wrench_reader: FixedJointWrenchReader | None = None
        self._hand_body_id: int | None = None
        self._ft_body_id: int | None = None
        self._reset_body_id_cache: dict[str, int] | None = None
        self._reset_collision_local_centers: torch.Tensor | None = None
        self._reset_collision_half_extents: torch.Tensor | None = None
        self._task_point_markers: VisualizationMarkers | None = None
        self._eef_frame_markers: VisualizationMarkers | None = None
        super().__init__(cfg=cfg, render_mode=render_mode)
        self._initialize_reset_collision_bounds()
        self._validate_sensor_contract()
        self.set_debug_vis(self.cfg.debug_vis)

    # ---------------------------------------------------------------------
    # Viewport visualization
    # ---------------------------------------------------------------------

    def set_debug_vis(self, enabled: bool) -> None:
        """Show or hide the task points and the virtual C/EEF frame.

        The markers are USD visuals only: they have no collision, mass, sensor,
        observation, or reward effect. All marker geometry is procedural so
        enabling this view does not add a Nucleus or reference-package asset
        dependency.
        """

        enabled = bool(enabled)
        self.cfg.debug_vis = enabled
        if enabled and self._task_point_markers is None:
            self._create_debug_markers()
        for marker in (self._task_point_markers, self._eef_frame_markers):
            if marker is not None:
                marker.set_visibility(enabled)
        if enabled:
            self._update_debug_markers()

    def _create_debug_markers(self) -> None:
        """Create package-local procedural markers for Target, Goal, and EEF."""

        self._task_point_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/BlindSweep/TaskPoints",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=0.035,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.30, 0.02),
                            emissive_color=(0.35, 0.06, 0.0),
                            opacity=0.40,
                        ),
                    ),
                    "goal": sim_utils.SphereCfg(
                        radius=0.030,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.05, 1.0, 0.15),
                            emissive_color=(0.01, 0.35, 0.03),
                            opacity=0.70,
                        ),
                    ),
                },
            )
        )

        axis_length = 0.080
        axis_thickness = 0.006
        self._eef_frame_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/BlindSweep/VirtualEEF",
                markers={
                    "origin": sim_utils.SphereCfg(
                        radius=0.009,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 1.0, 1.0),
                            emissive_color=(0.30, 0.30, 0.30),
                        ),
                    ),
                    "x_axis": sim_utils.CuboidCfg(
                        size=(axis_length, axis_thickness, axis_thickness),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.05, 0.05),
                            emissive_color=(0.30, 0.0, 0.0),
                        ),
                    ),
                    "y_axis": sim_utils.CuboidCfg(
                        size=(axis_length, axis_thickness, axis_thickness),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.05, 1.0, 0.05),
                            emissive_color=(0.0, 0.30, 0.0),
                        ),
                    ),
                    "z_axis": sim_utils.CuboidCfg(
                        size=(axis_length, axis_thickness, axis_thickness),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.05, 0.20, 1.0),
                            emissive_color=(0.0, 0.03, 0.30),
                        ),
                    ),
                },
            )
        )

    def _update_debug_markers(self) -> None:
        """Move markers to the live object, episode goal, and virtual EEF."""

        if self._task_point_markers is None or self._eef_frame_markers is None:
            return

        target = self.scene["target_object"]
        target_and_goal = torch.cat((target.data.root_pos_w, self.goal_pos_w), dim=0)
        point_indices = torch.cat(
            (
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs, dtype=torch.long, device=self.device),
            )
        )
        self._task_point_markers.visualize(
            translations=target_and_goal,
            marker_indices=point_indices,
        )

        c_pos_w, c_quat_w = self.control_point_pose_w()
        half_axis = 0.040
        local_axis_centers = torch.tensor(
            (
                (half_axis, 0.0, 0.0),
                (0.0, half_axis, 0.0),
                (0.0, 0.0, half_axis),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        axis_centers_w = [
            c_pos_w
            + math_utils.quat_apply(
                c_quat_w,
                local_axis_centers[axis].expand(self.num_envs, -1),
            )
            for axis in range(3)
        ]
        quarter_turn = math.sqrt(0.5)
        local_axis_quats = torch.tensor(
            (
                (1.0, 0.0, 0.0, 0.0),
                (quarter_turn, 0.0, 0.0, quarter_turn),
                (quarter_turn, 0.0, -quarter_turn, 0.0),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        axis_quats_w = [
            math_utils.quat_mul(
                c_quat_w,
                local_axis_quats[axis].expand(self.num_envs, -1),
            )
            for axis in range(3)
        ]
        eef_indices = torch.cat(
            [
                torch.full(
                    (self.num_envs,), index, dtype=torch.long, device=self.device
                )
                for index in range(4)
            ]
        )
        self._eef_frame_markers.visualize(
            translations=torch.cat((c_pos_w, *axis_centers_w), dim=0),
            orientations=torch.cat((c_quat_w, *axis_quats_w), dim=0),
            marker_indices=eef_indices,
        )

    # ---------------------------------------------------------------------
    # Episode specification and reset finalization
    # ---------------------------------------------------------------------

    def sample_episode_specs(self, env_ids: torch.Tensor) -> None:
        """Sample mode/command/object/initial C target once per episode.

        Rejection sampling enforces both fixed start and fixed goal inside the
        usable shelf rectangle.  As in Sweep-Policy, commands are equiprobable
        discrete shelf-left/shelf-right motions along world +/-Y only.
        """

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        task = self.cfg.task
        pending = torch.arange(len(env_ids), device=self.device)
        sampled_xy = torch.zeros((len(env_ids), 2), device=self.device)
        sampled_angle = torch.zeros(len(env_ids), device=self.device)
        sampled_distance = torch.zeros(len(env_ids), device=self.device)
        attempts = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        sampled_surface_mode = (
            torch.rand(len(env_ids), device=self.device) >= task.palm_mode_probability
        ).long()

        board_min = torch.tensor(
            (
                task.board_center[0] - 0.5 * task.board_size[0] + task.board_edge_margin,
                task.board_center[1] - 0.5 * task.board_size[1] + task.board_edge_margin,
            ),
            device=self.device,
        )
        board_max = torch.tensor(
            (
                task.board_center[0] + 0.5 * task.board_size[0] - task.board_edge_margin,
                task.board_center[1] + 0.5 * task.board_size[1] - task.board_edge_margin,
            ),
            device=self.device,
        )
        object_low = torch.tensor(task.object_xy_range_low, device=self.device)
        object_high = torch.tensor(task.object_xy_range_high, device=self.device)

        for _ in range(task.command_sample_attempts):
            if pending.numel() == 0:
                break
            attempts[pending] += 1
            angle = sample_sweep_angles(len(pending), device=self.device)
            distance = torch.empty(len(pending), device=self.device).uniform_(*task.command_distance_range)
            xy = object_low + torch.rand((len(pending), 2), device=self.device) * (object_high - object_low)
            direction_xy = direction_from_angle(angle)[:, :2]
            goal_xy = xy + distance.unsqueeze(-1) * direction_xy
            start_valid = torch.all((xy >= board_min) & (xy <= board_max), dim=-1)
            goal_valid = torch.all((goal_xy >= board_min) & (goal_xy <= board_max), dim=-1)
            # The active manipulator reset places C behind the Cube opposite
            # the commanded shelf-left/right motion. Reject object/direction
            # pairs whose complete longitudinal jitter envelope would put C
            # outside the reference's conservative shelf-y region. This is a
            # cheap episode-spec constraint, not online IK rejection.
            # Dorsal orientation reverses Hand +Y. Since C has a fixed +Y
            # offset from the hand center, preserve the same physical hand
            # standoff by moving dorsal C targets back by 2*|T_HC.y|.
            stable_backoff = (
                task.stable_reset_position_offset_task[0]
                + sampled_surface_mode[pending].to(dtype=xy.dtype)
                * (2.0 * abs(task.c_offset_h[1]))
            )
            stable_backoff_jitter = task.stable_reset_position_jitter_task[0]
            stable_c_y = xy[:, 1] - stable_backoff * direction_xy[:, 1]
            stable_c_y_valid = (
                stable_c_y - stable_backoff_jitter
                >= task.stable_reset_c_y_range[0]
            ) & (
                stable_c_y + stable_backoff_jitter
                <= task.stable_reset_c_y_range[1]
            )
            valid = start_valid & goal_valid
            stable_valid = valid & stable_c_y_valid
            accepted_rows = pending[stable_valid]
            sampled_xy[accepted_rows] = xy[stable_valid]
            sampled_angle[accepted_rows] = angle[stable_valid]
            sampled_distance[accepted_rows] = distance[stable_valid]
            pending = pending[~stable_valid]
        if pending.numel() > 0:
            raise RuntimeError(
                f"Unable to sample {len(pending)} board-valid commands after "
                f"{task.command_sample_attempts} attempts."
            )

        origin = (
            self.scene.env_origins[env_ids]
            if hasattr(self, "scene")
            else torch.zeros((len(env_ids), 3), device=self.device)
        )
        direction = direction_from_angle(sampled_angle)
        board_top = task.board_center[2] + 0.5 * task.board_size[2]
        object_pos_local = torch.cat(
            (
                sampled_xy,
                torch.full((len(env_ids), 1), board_top + 0.5 * task.cube_size, device=self.device),
            ),
            dim=-1,
        )
        object_pos_w = object_pos_local + origin

        self.surface_mode[env_ids] = sampled_surface_mode
        self.command_angle[env_ids] = sampled_angle
        self.command_distance[env_ids] = sampled_distance
        self.command_direction_w[env_ids] = direction
        self.object_initial_pos_w[env_ids] = object_pos_w
        self.object_initial_quat_w[env_ids] = torch.tensor(
            task.initial_cube_quat_w, device=self.device
        ).expand(len(env_ids), -1)
        self.goal_pos_w[env_ids] = object_pos_w + sampled_distance.unsqueeze(-1) * direction
        self.reset_spec_attempts[env_ids] = attempts

        tangent = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), dim=-1)
        longitudinal = torch.empty(len(env_ids), device=self.device).uniform_(*task.c_longitudinal_variation)
        lateral = torch.empty(len(env_ids), device=self.device).uniform_(*task.c_lateral_variation)
        height = torch.empty(len(env_ids), device=self.device).uniform_(*task.c_height_variation)
        self.desired_c_pos_w[env_ids] = (
            object_pos_w
            - (task.c_standoff + longitudinal).unsqueeze(-1) * direction
            + lateral.unsqueeze(-1) * tangent
        )
        self.desired_c_pos_w[env_ids, 2] += task.c_vertical_offset + height

        # H +Y is the palmar outward normal and H +X is the hand's up axis in
        # the package-local Inspire model (H +Z runs along the fingers).
        # Dorsal mode aligns H -Y with the command; never synthesize it as a
        # wrist-3 pi flip.
        mode_sign = torch.where(
            self.surface_mode[env_ids] == 0,
            torch.ones(len(env_ids), device=self.device),
            -torch.ones(len(env_ids), device=self.device),
        )
        y_axis = mode_sign.unsqueeze(-1) * direction
        x_axis = torch.zeros_like(y_axis)
        x_axis[:, 2] = 1.0
        z_axis = torch.linalg.cross(x_axis, y_axis, dim=-1)
        z_axis = torch.nn.functional.normalize(z_axis, dim=-1)
        y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
        nominal_rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
        nominal_quat = math_utils.quat_unique(math_utils.quat_from_matrix(nominal_rotation))
        rpy = torch.empty((len(env_ids), 3), device=self.device)
        rpy[:, 0].uniform_(*task.c_roll_variation)
        rpy[:, 1].uniform_(*task.c_pitch_variation)
        rpy[:, 2].uniform_(*task.c_yaw_variation)
        hand_quat = math_utils.quat_unique(
            math_utils.quat_mul(
                nominal_quat,
                math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2]),
            )
        )
        selected_normal_h = torch.zeros_like(direction)
        selected_normal_h[:, 1] = mode_sign
        selected_normal_w = math_utils.quat_apply(hand_quat, selected_normal_h)
        alignment = torch.sum(selected_normal_w * direction, dim=-1)
        if torch.any(alignment < task.surface_alignment_min_dot):
            failed = torch.where(alignment < task.surface_alignment_min_dot)[0].tolist()
            raise RuntimeError(
                "Configured orientation variation no longer points the selected hand surface "
                f"toward the cube for sampled rows {failed[:16]}"
            )
        c_quat_h = torch.tensor(task.c_quat_h, device=self.device).expand(len(env_ids), -1)
        self.desired_c_quat_w[env_ids] = math_utils.quat_unique(
            math_utils.quat_mul(hand_quat, c_quat_h)
        )

        hand_low = torch.tensor(task.initial_hand_open_range_low, device=self.device)
        hand_high = torch.tensor(task.initial_hand_open_range_high, device=self.device)
        self.initial_hand_synergy[env_ids] = hand_low + torch.rand((len(env_ids), 2), device=self.device) * (
            hand_high - hand_low
        )

    def _finalize_reset(self, env_ids: torch.Tensor) -> None:
        # Re-certify the final state after the chosen best-q was written and
        # the caller performed sim.forward()/scene.update().  This makes the
        # episode gate depend only on the state actually exposed to the policy.
        self.reset_collision_free[env_ids] = self.reset_collision_free_mask(env_ids)
        if not bool(self.reset_collision_free[env_ids].all()):
            bad_ids = env_ids[~self.reset_collision_free[env_ids]].tolist()
            raise RuntimeError(
                "Reset finalization received candidates that did not pass the "
                f"conservative live-FK collision certificate: {bad_ids[:16]}"
            )
        robot = self.scene["robot"]
        target = self.scene["target_object"]
        c_pos, c_quat = self.control_point_pose_w()
        self.c0_pos_w[env_ids] = c_pos[env_ids]
        self.c0_quat_w[env_ids] = math_utils.quat_unique(c_quat[env_ids])
        object_pos = target.data.root_pos_w[env_ids]
        object_quat = target.data.root_quat_w[env_ids]
        relative_pos, _ = math_utils.subtract_frame_transforms(
            self.c0_pos_w[env_ids],
            self.c0_quat_w[env_ids],
            object_pos,
            object_quat,
        )
        self.initial_object_pos_c0[env_ids] = relative_pos
        upright_local = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(len(env_ids), -1)
        self.initial_object_upright_w[env_ids] = math_utils.quat_apply(object_quat, upright_local)
        self.initial_object_height_w[env_ids] = object_pos[:, 2]
        self.object_initial_pos_w[env_ids] = object_pos
        self.object_initial_quat_w[env_ids] = object_quat
        arm_term = self.action_manager.get_term("arm_action")
        arm_joint_ids = torch.as_tensor(arm_term.joint_ids, dtype=torch.long, device=self.device)
        self.initial_arm_joint_pos[env_ids] = torch.index_select(
            robot.data.joint_pos[env_ids], 1, arm_joint_ids
        )
        hand_term = self.action_manager.get_term("hand_action")
        self.initial_hand_actual_synergy[env_ids] = hand_term.actual_synergy[env_ids]
        self.current_policy_action[env_ids] = 0.0
        self.last_policy_action[env_ids] = 0.0
        self.previous_policy_action[env_ids] = 0.0
        self.has_previous_policy_action[env_ids] = False
        # The conservative live-FK certificate established a contact-free
        # reset.  Mask stale backend tensors to zero until the first ordinary
        # physics substep; subsequent F/T samples are uncancelled and include
        # the Hand's self-weight force and moment.
        self.sensor_valid[env_ids] = True
        self.sensor_data_fresh[env_ids] = False
        self.invalid_reset_latched[env_ids] = False

        # Action terms reset after reset events; explicitly re-synchronize their
        # non-accumulating target state with the actual FK/joint state.
        for term in self.action_manager._terms.values():
            reset_to_current = getattr(term, "reset_to_current", None)
            if reset_to_current is not None:
                reset_to_current(env_ids)

        # Store reset accuracy for audit logs.
        self.reset_position_error[env_ids] = torch.linalg.vector_norm(
            c_pos[env_ids] - self.desired_c_pos_w[env_ids], dim=-1
        )
        _, rotation_error = math_utils.compute_pose_error(
            c_pos[env_ids],
            c_quat[env_ids],
            c_pos[env_ids],
            self.desired_c_quat_w[env_ids],
            rot_error_type="axis_angle",
        )
        self.reset_orientation_error[env_ids] = torch.linalg.vector_norm(rotation_error, dim=-1)
        robot.set_joint_velocity_target(
            torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids
        )

    def reset_workspace_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check the straight approach-to-goal C path against a reach shell.

        This inexpensive prefilter is deliberately conservative and does not
        replace full-pose IK at the sampled approach state.  Nine samples make
        the configured inner reach bound apply to the path interior as well as
        its endpoints.
        """

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        interpolation = torch.linspace(0.0, 1.0, 9, device=self.device).view(1, -1, 1)
        displacement = (
            self.command_distance[env_ids].unsqueeze(-1) * self.command_direction_w[env_ids]
        )
        path_w = self.desired_c_pos_w[env_ids].unsqueeze(1) + interpolation * displacement.unsqueeze(1)
        base_w = self.scene["robot"].data.root_link_pos_w[env_ids].unsqueeze(1)
        path_b = path_w - base_w
        reach = torch.linalg.vector_norm(path_b, dim=-1)
        height = path_b[..., 2]
        self.reset_path_min_reach_m[env_ids] = reach.amin(dim=-1)
        self.reset_path_max_reach_m[env_ids] = reach.amax(dim=-1)
        reach_low, reach_high = self.cfg.task.reset_workspace_reach_range_m
        height_low, height_high = self.cfg.task.reset_workspace_height_range_m
        return torch.all(
            (reach >= reach_low)
            & (reach <= reach_high)
            & (height >= height_low)
            & (height <= height_high),
            dim=-1,
        )

    def reset_clearance_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Conservatively validate robot--board/object separation on the GPU.

        PhysX contact tensors become current only after advancing physics,
        which would create an unlogged transition during a partial vector
        reset.  Instead, reset acceptance uses capsule samples for the long arm
        links and conservative spheres for every hand body.  All calculations
        use the just-updated articulation body poses and therefore require no
        ``sim.step``.  The radii are explicit bring-up proposals and the final
        contact-free rate remains a smoke-test acceptance metric.
        """

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        robot = self.scene["robot"]
        if self._reset_body_id_cache is None:
            required = set(ROBOT_CONTACT_BODY_NAMES)
            self._reset_body_id_cache = {}
            for name in required:
                ids, names = robot.find_bodies(name, preserve_order=True)
                if len(ids) != 1 or names[0] != name:
                    raise RuntimeError(f"Reset clearance body {name!r} did not resolve exactly once")
                self._reset_body_id_cache[name] = int(ids[0])

        ids_by_name = self._reset_body_id_cache
        body_pos = robot.data.body_pos_w
        point_groups: list[torch.Tensor] = []
        radius_groups: list[torch.Tensor] = []

        # Sample long links more densely than their radius so no gap exists
        # between neighboring conservative capsule samples.
        arm_capsules = (
            # Radii include joint housings and a conservative proxy error
            # allowance, not just the nominal cylindrical link radius.
            ("upper_arm_link", "forearm_link", 0.170),
            ("forearm_link", "wrist_1_link", 0.140),
            ("wrist_1_link", "wrist_2_link", 0.090),
            ("wrist_2_link", "wrist_3_link", 0.085),
            ("wrist_3_link", "axia80_link", 0.060),
            ("axia80_link", "inspire_base_link", 0.060),
        )
        interpolation = torch.linspace(0.0, 1.0, 11, device=self.device).view(1, -1, 1)
        for start_name, end_name, radius in arm_capsules:
            start = body_pos[env_ids, ids_by_name[start_name]].unsqueeze(1)
            end = body_pos[env_ids, ids_by_name[end_name]].unsqueeze(1)
            points = start + interpolation * (end - start)
            point_groups.append(points)
            radius_groups.append(torch.full((points.shape[1],), radius, device=self.device))

        discrete_names = ("base_link", "shoulder_link", *HAND_CONTACT_BODY_NAMES)
        discrete_points = torch.stack(
            [body_pos[env_ids, ids_by_name[name]] for name in discrete_names], dim=1
        )
        discrete_radii = torch.tensor(
            [
                0.120 if name in ("base_link", "shoulder_link")
                else 0.080 if name == HAND_BASE_BODY_NAME
                else 0.018 if name in PALM_SENSOR_BODY_NAMES
                else 0.035
                for name in discrete_names
            ],
            device=self.device,
        )
        point_groups.append(discrete_points)
        radius_groups.append(discrete_radii)

        points_w = torch.cat(point_groups, dim=1)
        radii = torch.cat(radius_groups).view(1, -1)
        margin = self.cfg.task.reset_clearance_margin_m

        board_center_w = self.scene.env_origins[env_ids] + torch.tensor(
            self.cfg.task.board_center, device=self.device
        )
        board_half = 0.5 * torch.tensor(self.cfg.task.board_size, device=self.device)
        board_delta = torch.clamp(
            torch.abs(points_w - board_center_w.unsqueeze(1)) - board_half,
            min=0.0,
        )
        board_clearance = torch.linalg.vector_norm(board_delta, dim=-1) - radii

        target = self.scene["target_object"]
        object_pos = target.data.root_pos_w[env_ids]
        object_quat = target.data.root_quat_w[env_ids]
        point_count = points_w.shape[1]
        object_relative = math_utils.quat_apply_inverse(
            object_quat.unsqueeze(1).expand(-1, point_count, -1).reshape(-1, 4),
            (points_w - object_pos.unsqueeze(1)).reshape(-1, 3),
        ).reshape(len(env_ids), point_count, 3)
        object_half = 0.5 * self.cfg.task.cube_size
        object_delta = torch.clamp(torch.abs(object_relative) - object_half, min=0.0)
        object_clearance = torch.linalg.vector_norm(object_delta, dim=-1) - radii

        minimum_board = board_clearance.amin(dim=-1)
        minimum_object = object_clearance.amin(dim=-1)
        minimum = torch.minimum(minimum_board, minimum_object)
        self.reset_min_board_clearance_m[env_ids] = minimum_board
        self.reset_min_object_clearance_m[env_ids] = minimum_object
        self.reset_min_clearance_m[env_ids] = minimum
        return minimum > margin

    def _initialize_reset_collision_bounds(self) -> None:
        """Cache a conservative body-local box around every robot collider.

        PhysX scene-query broadphase poses are stale immediately after tensor
        joint teleports, even after ``sim.forward()``.  Reset certification
        therefore cannot query the live articulation through overlap_shape.
        Instead, this method encloses every authored collision geometry in a
        body-local AABB once.  Live link poses turn those boxes into world OBBs
        during IK, without a hidden positive-time simulation step.
        """

        from isaaclab.sim.utils import get_current_stage
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = get_current_stage()
        robot = self.scene["robot"]
        robot_root = f"{self.scene.env_prim_paths[0]}/Robot"
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        local_centers: list[tuple[float, float, float]] = []
        half_extents: list[tuple[float, float, float]] = []

        for body_name in robot.body_names:
            body = stage.GetPrimAtPath(f"{robot_root}/{body_name}")
            if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing reset-collision rigid body {body.GetPath()}")
            body_inverse = xform_cache.GetLocalToWorldTransform(body).GetInverse()
            shape_paths: set[str] = set()
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                for shape in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if UsdGeom.Boundable(shape):
                        shape_paths.add(str(shape.GetPath()))

            points_body: list[Gf.Vec3d] = []
            for shape_path in sorted(shape_paths):
                shape = stage.GetPrimAtPath(shape_path)
                boundable = UsdGeom.Boundable(shape)
                # Compute from geometry attributes instead of reading the
                # authored extent.  Procedural prims may retain USD's fallback
                # [-1, 1]^3 extent (the Axia80 Cylinder is one such case).
                extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
                    boundable, Usd.TimeCode.Default()
                )
                if extent is None or len(extent) != 2:
                    raise RuntimeError(f"Collider {shape_path} has no computable extent")
                shape_to_world = xform_cache.GetLocalToWorldTransform(shape)
                for x in (float(extent[0][0]), float(extent[1][0])):
                    for y in (float(extent[0][1]), float(extent[1][1])):
                        for z in (float(extent[0][2]), float(extent[1][2])):
                            point_world = shape_to_world.Transform(Gf.Vec3d(x, y, z))
                            points_body.append(body_inverse.Transform(point_world))

            if not points_body:
                raise RuntimeError(f"Rigid body {body.GetPath()} has no collision geometry")
            minimum = tuple(min(float(point[axis]) for point in points_body) for axis in range(3))
            maximum = tuple(max(float(point[axis]) for point in points_body) for axis in range(3))
            local_centers.append(
                tuple(0.5 * (minimum[axis] + maximum[axis]) for axis in range(3))
            )
            half_extents.append(
                tuple(0.5 * (maximum[axis] - minimum[axis]) for axis in range(3))
            )

        self._reset_collision_local_centers = torch.tensor(
            local_centers, dtype=torch.float32, device=self.device
        )
        self._reset_collision_half_extents = torch.tensor(
            half_extents, dtype=torch.float32, device=self.device
        )
        if (
            not torch.isfinite(self._reset_collision_local_centers).all()
            or not torch.isfinite(self._reset_collision_half_extents).all()
            or torch.any(self._reset_collision_half_extents <= 0.0)
        ):
            raise RuntimeError("Reset collision bounds are non-finite or degenerate")
        self._reset_last_cube_overlap = torch.zeros(
            (self.num_envs, len(robot.body_names)), dtype=torch.bool, device=self.device
        )
        self._reset_last_board_overlap = torch.zeros_like(self._reset_last_cube_overlap)

    def reset_collision_free_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Certify Robot--Cube/Board disjointness from current FK body poses.

        Each live robot OBB encloses its complete collision mesh.  A full
        15-axis SAT test against the exact Cube and Board boxes is therefore
        conservative: a returned ``True`` cannot hide a mesh collision, while
        false positives merely cause the reset candidate to be resampled.
        """

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        if self._reset_collision_local_centers is None or self._reset_collision_half_extents is None:
            raise RuntimeError("Reset collision bounds were not initialized")

        robot = self.scene["robot"]
        body_position_w = robot.data.body_pos_w[env_ids]
        body_quaternion_w = robot.data.body_quat_w[env_ids]
        environment_count, body_count = body_position_w.shape[:2]
        local_centers = self._reset_collision_local_centers.expand(environment_count, -1, -1)
        body_center_w = body_position_w + math_utils.quat_apply(
            body_quaternion_w.reshape(-1, 4), local_centers.reshape(-1, 3)
        ).reshape(environment_count, body_count, 3)
        body_rotation_w = math_utils.matrix_from_quat(body_quaternion_w)
        body_half_extent = self._reset_collision_half_extents.unsqueeze(0)
        body_half_extent = body_half_extent + self.cfg.task.reset_clearance_margin_m

        target = self.scene["target_object"]
        cube_center_w = target.data.root_pos_w[env_ids].unsqueeze(1)
        cube_rotation_w = math_utils.matrix_from_quat(
            target.data.root_quat_w[env_ids]
        ).unsqueeze(1)
        cube_half_extent = torch.full(
            (1, 1, 3),
            0.5 * self.cfg.task.cube_size,
            dtype=body_center_w.dtype,
            device=self.device,
        )
        cube_overlap = oriented_box_overlap(
            body_center_w,
            body_rotation_w,
            body_half_extent,
            cube_center_w,
            cube_rotation_w,
            cube_half_extent,
        )

        board = self.scene["board"]
        board_center_w = board.data.root_pos_w[env_ids]
        board_rotation_w = math_utils.matrix_from_quat(
            board.data.root_quat_w[env_ids]
        ).unsqueeze(1)
        board_half_extent = (
            0.5
            * torch.tensor(
                self.cfg.task.board_size,
                dtype=body_center_w.dtype,
                device=self.device,
            )
        ).view(1, 1, 3)
        board_overlap = oriented_box_overlap(
            body_center_w,
            body_rotation_w,
            body_half_extent,
            board_center_w.unsqueeze(1),
            board_rotation_w,
            board_half_extent,
        )
        self._reset_last_cube_overlap[env_ids] = cube_overlap
        self._reset_last_board_overlap[env_ids] = board_overlap
        return ~(cube_overlap.any(dim=-1) | board_overlap.any(dim=-1))

    # ---------------------------------------------------------------------
    # Shared sensor/kinematics accessors
    # ---------------------------------------------------------------------

    def _resolve_hand_body(self) -> int:
        if self._hand_body_id is None:
            ids, names = self.scene["robot"].find_bodies(HAND_BASE_BODY_NAME, preserve_order=True)
            if len(ids) != 1:
                raise RuntimeError(f"Expected one hand base body, got {names}")
            self._hand_body_id = int(ids[0])
        return self._hand_body_id

    def _resolve_ft_body(self) -> int:
        if self._ft_body_id is None:
            ids, names = self.scene["robot"].find_bodies(FT_SENSOR_BODY_NAME, preserve_order=True)
            if len(ids) != 1:
                raise RuntimeError(f"Expected one Axia80 F body, got {names}")
            self._ft_body_id = int(ids[0])
        return self._ft_body_id

    def _validate_sensor_contract(self) -> None:
        """Validate body/channel order and supported one-to-many sensor shapes."""

        palm = self.scene["palm_tactile"]
        dorsal = self.scene["dorsal_tactile"]
        if set(palm.body_names) != set(PALM_SENSOR_BODY_NAMES) or len(palm.body_names) != 17:
            raise RuntimeError(f"Palm sensor must resolve the exact 17 channels, got {palm.body_names}")
        missing_dorsal = set(DORSAL_PARENT_BODY_NAMES) - set(dorsal.body_names)
        if missing_dorsal:
            raise RuntimeError(f"Dorsal approximation is missing parent bodies: {sorted(missing_dorsal)}")

        expected_filters = {
            "board_contacts": len(ROBOT_CONTACT_BODY_NAMES),
            "target_hand_contacts": len(HAND_CONTACT_BODY_NAMES),
            "target_robot_contacts": len(ROBOT_CONTACT_BODY_NAMES),
        }
        for sensor_name, filter_count in expected_filters.items():
            sensor = self.scene[sensor_name]
            matrix = sensor.data.force_matrix_w
            if len(sensor.body_names) != 1 or matrix is None or matrix.ndim != 4:
                raise RuntimeError(
                    f"{sensor_name} must be a one-body filtered contact sensor; "
                    f"bodies={sensor.body_names}, matrix={None if matrix is None else tuple(matrix.shape)}"
                )
            if matrix.shape[1] != 1 or matrix.shape[2] != filter_count:
                raise RuntimeError(
                    f"{sensor_name} expected (N, 1, {filter_count}, 3), got {tuple(matrix.shape)}"
                )

        robot_contacts = self.scene["robot_contacts"]
        if set(robot_contacts.body_names) != set(ROBOT_CONTACT_BODY_NAMES):
            raise RuntimeError(
                "Reset contact sensor did not resolve the complete arm/Axia80/hand body set: "
                f"{robot_contacts.body_names}"
            )
        # Resolve both frames and the incoming-wrench child eagerly.
        self._resolve_hand_body()
        self._resolve_ft_body()
        reader = self._get_wrench_reader()
        if reader.child_body_name != FT_MEASUREMENT_CHILD_BODY_NAME:
            raise RuntimeError("Wrench reader is not indexed by the measurement-joint child body")

    def control_point_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        body_id = self._resolve_hand_body()
        offset = torch.tensor(self.cfg.task.c_offset_h, device=self.device).expand(self.num_envs, -1)
        offset_quat = torch.tensor(self.cfg.task.c_quat_h, device=self.device).expand(self.num_envs, -1)
        return math_utils.combine_frame_transforms(
            robot.data.body_pos_w[:, body_id], robot.data.body_quat_w[:, body_id], offset, offset_quat
        )

    def tactile_sample(self):
        if self._tactile_reader is None:
            self._tactile_reader = BilateralTactileReader(
                self.scene["palm_tactile"],
                self.scene["dorsal_tactile"],
                threshold_n=self.cfg.task.tactile_threshold_n,
                dorsal_threshold_n=self.cfg.task.dorsal_tactile_threshold_n,
            )
        return self._tactile_reader.read()

    def selected_tactile_bits(self) -> torch.Tensor:
        sample = self.tactile_sample()
        return torch.where(
            self.surface_mode.unsqueeze(-1).bool(), sample.dorsal_bits, sample.palm_bits
        ).to(dtype=torch.float32)

    def _get_wrench_reader(self) -> FixedJointWrenchReader:
        if self._wrench_reader is None:
            self._wrench_reader = FixedJointWrenchReader(
                self.scene["robot"],
                FT_MEASUREMENT_CHILD_BODY_NAME,
            )
        return self._wrench_reader

    def wrist_wrench_c(self):
        robot = self.scene["robot"]
        body_id = self._resolve_ft_body()
        f_pos = robot.data.body_pos_w[:, body_id]
        f_quat = robot.data.body_quat_w[:, body_id]
        c_pos, c_quat = self.control_point_pose_w()
        r_c_to_f_c = math_utils.quat_apply_inverse(c_quat, f_pos - c_pos)
        rotation_w_f = math_utils.matrix_from_quat(f_quat)
        rotation_w_c = math_utils.matrix_from_quat(c_quat)
        rotation_c_from_f = rotation_w_c.transpose(-1, -2) @ rotation_w_f
        return self._get_wrench_reader().read(rotation_c_from_f, r_c_to_f_c)

    def hand_object_normal(self):
        matrix = self.scene["target_hand_contacts"].data.force_matrix_w
        if matrix is None:
            raise RuntimeError("Target hand-contact sensor requires robot-hand filter paths.")
        # The sensor body is the object, so force_matrix_w already expresses
        # the normal force on the object; do not negate it.
        return hand_object_resultant_normal(matrix, input_forces_on_hand=False)

    # ---------------------------------------------------------------------
    # Physics-substep safety update and RL step
    # ---------------------------------------------------------------------

    def _update_substep_state(self) -> None:
        target = self.scene["target_object"]
        self.episode_physics_substep_count += 1
        # scene.update() has just refreshed the live contact and incoming-joint
        # tensors for this ordinary action-bearing physics substep.
        self.sensor_data_fresh[self.sensor_valid] = True
        tactile = self.tactile_sample()
        selected_bits = torch.where(
            self.surface_mode.unsqueeze(-1).bool(), tactile.dorsal_bits, tactile.palm_bits
        )
        selected_contact = torch.any(selected_bits.bool(), dim=-1)
        first_contact = (self.first_selected_contact_substep < 0) & selected_contact
        self.first_selected_contact_substep[first_contact] = self.episode_physics_substep_count[
            first_contact
        ]
        self.selected_contact_loss_count += (
            self.selected_contact_previous & ~selected_contact
        ).long()
        self.selected_contact_previous[:] = selected_contact

        board_sample = robot_board_contact_forces(self.scene["board_contacts"])
        self.board_force[:] = board_sample.total_magnitude_n
        self.board_force_peak[:] = torch.maximum(self.board_force_peak, self.board_force)
        link_force, link_index = board_sample.per_link_magnitudes_n.max(dim=-1)
        new_link_peak = link_force > self.board_link_force_peak
        self.board_link_force_peak[new_link_peak] = link_force[new_link_peak]
        self.board_link_index_peak[new_link_peak] = link_index[new_link_peak]
        if new_link_peak.any():
            contact_body_ids = []
            for name in ROBOT_CONTACT_BODY_NAMES:
                ids, names = self.scene["robot"].find_bodies(name, preserve_order=True)
                if len(ids) != 1 or names[0] != name:
                    raise RuntimeError(f"Board-contact body {name!r} did not resolve exactly once")
                contact_body_ids.append(int(ids[0]))
            contact_body_ids_tensor = torch.tensor(contact_body_ids, device=self.device)
            selected_body_ids = contact_body_ids_tensor[link_index]
            rows = torch.arange(self.num_envs, device=self.device)
            body_positions = self.scene["robot"].data.body_pos_w[rows, selected_body_ids]
            self.board_link_position_peak_w[new_link_peak] = body_positions[new_link_peak]
            c_pos, _ = self.control_point_pose_w()
            self.board_contact_c_position_peak_w[new_link_peak] = c_pos[new_link_peak]

        upright_local = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(self.num_envs, -1)
        upright = math_utils.quat_apply(target.data.root_quat_w, upright_local)
        self.current_tilt[:] = upright_tilt_angle(upright, self.initial_object_upright_w)
        self.tilt_peak[:] = torch.maximum(self.tilt_peak, self.current_tilt)
        self.current_height_increase[:] = target.data.root_pos_w[:, 2] - self.initial_object_height_w
        self.height_peak[:] = torch.maximum(self.height_peak, self.current_height_increase)

        self.board_hard_latched |= self.board_force > self.cfg.task.board_force_hard_n
        self.tilt_hard_latched |= self.current_tilt > self.cfg.task.tilt_hard_rad
        self.height_hard_latched |= self.current_height_increase > self.cfg.task.height_hard_m

        finite = (
            torch.isfinite(target.data.root_state_w).all(dim=-1)
            & torch.isfinite(self.scene["robot"].data.joint_pos).all(dim=-1)
            & torch.isfinite(self.scene["robot"].data.joint_vel).all(dim=-1)
        )
        self.invalid_sim_latched |= ~finite

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.extras["log"] = {}
        action = action.to(self.device)
        if action.shape != (self.num_envs, 8):
            raise ValueError(f"Expected policy action shape {(self.num_envs, 8)}, got {tuple(action.shape)}")
        sanitized_action = torch.where(
            torch.isfinite(action), action, torch.zeros_like(action)
        )
        bounded_action = torch.clamp(sanitized_action, -1.0, 1.0)
        self.last_policy_action[:] = bounded_action
        # Reset overlap validation and the initialized zero sensor buffer are
        # complete before an observation is returned.  Every caller action is
        # therefore processed once here and applied for the full decimation
        # window; ActionManager history and recorder data stay authoritative.
        self.action_manager.process_action(bounded_action)
        arm_executed = self.action_manager.get_term("arm_action").raw_actions
        hand_executed = self.action_manager.get_term("hand_action").processed_actions
        self.current_policy_action[:] = torch.cat((arm_executed, hand_executed), dim=-1)
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._update_substep_state()

        arm_term = self.action_manager.get_term("arm_action")
        hand_term = self.action_manager.get_term("hand_action")
        self.arm_torque_saturation_steps += arm_term.torque_saturated.long()
        self.hand_action_saturation_steps += hand_term.action_saturated.long()

        if self.invalid_sim_latched.any():
            invalid_ids = self.invalid_sim_latched.nonzero(as_tuple=False).squeeze(-1)[:16].tolist()
            self.failure_reason[self.invalid_sim_latched] = int(TerminationReason.INVALID_SIMULATION)
            self.last_termination_reason[self.invalid_sim_latched] = int(
                TerminationReason.INVALID_SIMULATION
            )
            self.extras["invalid_simulation"] = {
                "env_ids": invalid_ids,
                "object_state_w": self.scene["target_object"].data.root_state_w[
                    self.invalid_sim_latched
                ].clone(),
                "robot_joint_position": self.scene["robot"].data.joint_pos[
                    self.invalid_sim_latched
                ].clone(),
                "robot_joint_velocity": self.scene["robot"].data.joint_vel[
                    self.invalid_sim_latched
                ].clone(),
            }
            raise FloatingPointError(f"Invalid simulation state in environments {invalid_ids}")

        # A failed reset certificate must never become a PPO sample.
        if self.invalid_reset_latched.any():
            invalid_reset_ids = self.invalid_reset_latched.nonzero(
                as_tuple=False
            ).squeeze(-1)[:16].tolist()
            raise RuntimeError(
                "Invalid reset state before reward computation in "
                f"environments {invalid_reset_ids}"
            )

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        # A terminal condition always wins over a simultaneous time limit.
        self.reset_time_outs = self.termination_manager.time_outs & ~self.reset_terminated
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self._classify_termination_reason()
        self.last_termination_reason[:] = torch.where(
            self.reset_buf,
            self.failure_reason,
            torch.zeros_like(self.failure_reason),
        )
        self.extras["termination_reason"] = self.last_termination_reason.clone()
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        c_position_w, c_quaternion_w = self.control_point_pose_w()
        desired_c_position_w, desired_c_quaternion_w = math_utils.combine_frame_transforms(
            self.scene["robot"].data.root_link_pos_w,
            self.scene["robot"].data.root_link_quat_w,
            arm_term.desired_c_pose_b[:, :3],
            arm_term.desired_c_pose_b[:, 3:],
        )
        arm_joint_ids = torch.as_tensor(
            arm_term.joint_ids, dtype=torch.long, device=self.device
        )
        arm_joint_velocity = torch.index_select(
            self.scene["robot"].data.joint_vel, 1, arm_joint_ids
        )
        arm_joint_position = torch.index_select(
            self.scene["robot"].data.joint_pos, 1, arm_joint_ids
        )
        target = self.scene["target_object"]
        object_displacement_w = target.data.root_pos_w - self.object_initial_pos_w
        object_goal_error_m = torch.linalg.vector_norm(
            target.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        c_displacement_w = c_position_w - self.c0_pos_w
        tactile = self.tactile_sample()
        selected_tactile_bits = torch.where(
            self.surface_mode.unsqueeze(-1).bool(), tactile.dorsal_bits, tactile.palm_bits
        )
        nonselected_tactile_bits = torch.where(
            self.surface_mode.unsqueeze(-1).bool(), tactile.palm_bits, tactile.dorsal_bits
        )
        wrench = self.wrist_wrench_c()
        first_contact_time_s = torch.where(
            self.first_selected_contact_substep >= 0,
            self.first_selected_contact_substep.float() * self.physics_dt,
            torch.full((self.num_envs,), -1.0, device=self.device),
        )
        reward_weighted_rate = {
            name: self.reward_manager._step_reward[:, index].clone()
            for index, name in enumerate(self.reward_manager.active_terms)
        }
        reward_raw: dict[str, torch.Tensor] = {}
        for name in self.reward_manager.active_terms:
            weight = self.reward_manager.get_term_cfg(name).weight
            reward_raw[name] = (
                reward_weighted_rate[name] / weight
                if weight != 0.0
                else torch.zeros_like(reward_weighted_rate[name])
            )
        reward_integrated = {
            name: value * self.step_dt for name, value in reward_weighted_rate.items()
        }
        training_log = {
            "Episode/surface_dorsal": self.surface_mode.float().clone(),
            "Episode/command_angle_rad": self.command_angle.clone(),
            "Episode/command_distance_m": self.command_distance.clone(),
            "Task/goal_error_m": object_goal_error_m.clone(),
            "Task/object_displacement_m": torch.linalg.vector_norm(
                object_displacement_w, dim=-1
            ).clone(),
            "Task/episode_length_steps": self.episode_length_buf.float().clone(),
            "Contact/selected_channel_count": selected_tactile_bits.float().sum(dim=-1),
            "Contact/nonselected_channel_count": nonselected_tactile_bits.float().sum(dim=-1),
            "Contact/first_selected_contact_time_s": first_contact_time_s.clone(),
            "Contact/selected_contact_loss_count": self.selected_contact_loss_count.float().clone(),
            "Wrench/measured_force_norm_n": torch.linalg.vector_norm(
                wrench.measured_c[:, :3], dim=-1
            ),
            "Wrench/measured_torque_norm_nm": torch.linalg.vector_norm(
                wrench.measured_c[:, 3:], dim=-1
            ),
            "Safety/board_force_peak_n": self.board_force_peak.clone(),
            "Safety/tilt_peak_rad": self.tilt_peak.clone(),
            "Safety/height_peak_m": self.height_peak.clone(),
            "Control/arm_torque_saturated": arm_term.torque_saturated.float().clone(),
            "Control/hand_action_saturated": hand_term.action_saturated.float().clone(),
            "Reset/ik_position_error_m": self.reset_position_error.clone(),
            "Reset/ik_orientation_error_rad": self.reset_orientation_error.clone(),
            "Reset/episode_spec_attempts": self.reset_episode_spec_attempts.float().clone(),
            "Reset/valid_ik_candidates": self.reset_valid_candidate_count.float().clone(),
            "Reset/rejection_count": self.reset_rejection_counts.sum(dim=-1).float().clone(),
            "Reset/selected_solution_cost": self.reset_selected_solution_cost.clone(),
            "Reset/wrist_3_rad": self.initial_arm_joint_pos[:, 5].clone(),
            "Reset/collision_free": self.reset_collision_free.float().clone(),
            "Termination/reason": self.last_termination_reason.float().clone(),
        }
        for name in self.reward_manager.active_terms:
            training_log[f"RewardRaw/{name}"] = reward_raw[name]
            training_log[f"RewardRate/{name}"] = reward_weighted_rate[name]
            training_log[f"RewardIntegrated/{name}"] = reward_integrated[name]
        for reason in TerminationReason:
            if reason is not TerminationReason.NONE:
                training_log[f"Termination/{reason.name.lower()}"] = (
                    self.last_termination_reason == int(reason)
                ).float()
        self.extras["episode_diagnostics"] = {
            "surface_mode": self.surface_mode.clone(),
            "command_angle_rad": self.command_angle.clone(),
            "command_distance_m": self.command_distance.clone(),
            "command_direction_w": self.command_direction_w.clone(),
            "goal_position_w": self.goal_pos_w.clone(),
            "initial_object_position_w": self.object_initial_pos_w.clone(),
            "initial_object_quaternion_w": self.object_initial_quat_w.clone(),
            "initial_c_position_w": self.c0_pos_w.clone(),
            "initial_c_quaternion_w": self.c0_quat_w.clone(),
            "initial_arm_joint_position_rad": self.initial_arm_joint_pos.clone(),
            "initial_hand_synergy": self.initial_hand_actual_synergy.clone(),
            "reset_ik_position_error_m": self.reset_position_error.clone(),
            "reset_ik_orientation_error_rad": self.reset_orientation_error.clone(),
            "reset_ik_seed_attempts": self.reset_ik_attempts.clone(),
            "reset_episode_spec_attempts": self.reset_episode_spec_attempts.clone(),
            "reset_command_sample_attempts": self.reset_spec_attempts.clone(),
            "reset_rejection_code": self.reset_rejection_code.clone(),
            "reset_rejection_counts": self.reset_rejection_counts.clone(),
            "reset_valid_candidate_count": self.reset_valid_candidate_count.clone(),
            "reset_selected_solution_cost": self.reset_selected_solution_cost.clone(),
            "reset_path_min_reach_m": self.reset_path_min_reach_m.clone(),
            "reset_path_max_reach_m": self.reset_path_max_reach_m.clone(),
            "reset_min_clearance_m": self.reset_min_clearance_m.clone(),
            "reset_min_board_clearance_m": self.reset_min_board_clearance_m.clone(),
            "reset_min_object_clearance_m": self.reset_min_object_clearance_m.clone(),
            "reset_collision_free": self.reset_collision_free.clone(),
            "object_position_w": target.data.root_pos_w.clone(),
            "object_displacement_w": object_displacement_w.clone(),
            "object_displacement_m": torch.linalg.vector_norm(
                object_displacement_w, dim=-1
            ).clone(),
            "object_goal_error_m": object_goal_error_m.clone(),
            "c_position_w": c_position_w.clone(),
            "c_quaternion_w": c_quaternion_w.clone(),
            "c_displacement_w": c_displacement_w.clone(),
            "c_displacement_m": torch.linalg.vector_norm(c_displacement_w, dim=-1).clone(),
            "first_selected_contact_time_s": first_contact_time_s.clone(),
            "selected_contact_loss_count": self.selected_contact_loss_count.clone(),
            "palm_tactile_magnitude_n": tactile.palm_magnitudes_n.clone(),
            "dorsal_tactile_magnitude_n": tactile.dorsal_magnitudes_n.clone(),
            "palm_tactile_bits": tactile.palm_bits.clone(),
            "dorsal_tactile_bits": tactile.dorsal_bits.clone(),
            "selected_tactile_bits": selected_tactile_bits.clone(),
            "nonselected_tactile_bits": nonselected_tactile_bits.clone(),
            "wrist_wrench_raw_f": wrench.raw_f.clone(),
            "wrist_wrench_measured_c": wrench.measured_c.clone(),
            "board_force_n": self.board_force.clone(),
            "board_force_peak_n": self.board_force_peak.clone(),
            "board_link_force_n": self.board_link_force_peak.clone(),
            "board_link_index": self.board_link_index_peak.clone(),
            "board_link_position_w": self.board_link_position_peak_w.clone(),
            "board_contact_c_position_w": self.board_contact_c_position_peak_w.clone(),
            "tilt_rad": self.current_tilt.clone(),
            "tilt_peak_rad": self.tilt_peak.clone(),
            "height_increase_m": self.current_height_increase.clone(),
            "height_peak_m": self.height_peak.clone(),
            "topple_hard_latched": self.tilt_hard_latched.clone(),
            "board_hard_latched": self.board_hard_latched.clone(),
            "height_hard_latched": self.height_hard_latched.clone(),
            "desired_c_position_w": desired_c_position_w.clone(),
            "desired_c_quaternion_w": desired_c_quaternion_w.clone(),
            "measured_c_twist_b": arm_term.c_twist_b.clone(),
            "jacobian_c_twist_at_control_b": arm_term.jacobian_c_twist_b.clone(),
            "arm_joint_position_rad": arm_joint_position.clone(),
            "arm_joint_effort_nm": arm_term.joint_efforts.clone(),
            "arm_torque_saturated": arm_term.torque_saturated.clone(),
            "arm_torque_saturation_steps": self.arm_torque_saturation_steps.clone(),
            "arm_joint_velocity_rad_s": arm_joint_velocity.clone(),
            "hand_actual_synergy": hand_term.actual_synergy.clone(),
            "hand_action_saturated": hand_term.action_saturated.clone(),
            "hand_action_saturation_steps": self.hand_action_saturation_steps.clone(),
            "current_executed_action": self.current_policy_action.clone(),
            "episode_length_steps": self.episode_length_buf.clone(),
            "termination_reason": self.last_termination_reason.clone(),
            "reward_raw": reward_raw,
            "reward_weighted_rate": reward_weighted_rate,
            "reward_integrated": reward_integrated,
        }

        self.previous_policy_action[:] = self.current_policy_action
        self.has_previous_policy_action[:] = True

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # RSL-RL persists only flat scalar tensors under ``extras['log']``.
        # Keep the full per-environment audit dictionary separately while also
        # exposing a compact, TensorBoard-compatible training summary.
        self.extras.setdefault("log", {}).update(training_log)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if self.cfg.debug_vis:
            self._update_debug_markers()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _classify_termination_reason(self) -> None:
        """Store one deterministic reason with failure > success > timeout."""

        target_pos = self.scene["target_object"].data.root_pos_w
        outside = object_out_of_bounds(self)
        task_failed = self.tilt_hard_latched | self.board_hard_latched | self.height_hard_latched | outside
        success = success_from_positions(
            target_pos,
            self.goal_pos_w,
            distance_threshold=self.cfg.task.success_distance_m,
        ) & ~task_failed & ~self.invalid_reset_latched
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        reason = torch.where(
            self.reset_time_outs,
            torch.full_like(reason, int(TerminationReason.TIMEOUT)),
            reason,
        )
        reason = torch.where(success, torch.full_like(reason, int(TerminationReason.SUCCESS)), reason)
        reason = torch.where(outside, torch.full_like(reason, int(TerminationReason.OUT_OF_BOUNDS)), reason)
        reason = torch.where(
            self.height_hard_latched,
            torch.full_like(reason, int(TerminationReason.HEIGHT_LIMIT)),
            reason,
        )
        reason = torch.where(
            self.board_hard_latched,
            torch.full_like(reason, int(TerminationReason.BOARD_CONTACT)),
            reason,
        )
        reason = torch.where(
            self.tilt_hard_latched,
            torch.full_like(reason, int(TerminationReason.TOPPLE)),
            reason,
        )
        reason = torch.where(
            self.invalid_reset_latched,
            torch.full_like(reason, int(TerminationReason.INVALID_RESET)),
            reason,
        )
        self.failure_reason[:] = reason

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.curriculum_manager.compute(env_ids=env_ids)
        self.sample_episode_specs(env_ids)
        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        self.extras["log"] = {}
        for manager in (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
            self.recorder_manager,
        ):
            self.extras["log"].update(manager.reset(env_ids))

        self.sim.forward()
        self.scene.update(dt=self.physics_dt)
        self._finalize_reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        if self.cfg.debug_vis:
            self._update_debug_markers()


__all__ = ["BlindSweepEnv"]
