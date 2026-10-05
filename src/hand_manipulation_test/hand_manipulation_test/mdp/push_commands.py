"""Reset-fixed pushing commands, physical state snapshots, and target markers."""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from ..action_math import quaternion_rotate
from ..contact_sensors import table_contact_mask
from ..geometry import base_link_pose_w
from ..push_state import PushSnapshot, PushStateConfig, PushStateTracker
from .contact_commands import CubeInitialPositionCommand, CubeInitialPositionCommandCfg


class CubePushCommand(CubeInitialPositionCommand):
    """Sample before reset IK; snapshot and initialize histories after reset IK.

    The reset event samples the complete specification once. CommandManager's
    later reset snapshots the realized Cube/EEF state without sampling again.
    Rewards and terminations share one cached state transition per policy step.
    """

    cfg: "CubePushCommandCfg"

    def __init__(self, cfg, env):
        self._goal_marker = None
        self._direction_marker = None
        super().__init__(cfg, env)
        # Reaching metrics refer to the initial Cube, whereas this task tracks
        # actual Cube-to-goal distance and stable pushing success.
        self.metrics.pop("distance_m", None)
        self.metrics.pop("reached", None)
        self.sampled_cube_pos_w = self.target_pos_w.clone()
        self.goal_pos_w = self.target_pos_w.clone()
        self.angle_rad = torch.zeros(self.num_envs, device=self.device)
        self.distance_m = torch.full_like(self.angle_rad, env.cfg.task.command_distance_range[0])
        self.direction_w = torch.zeros_like(self.target_pos_w)
        self.direction_w[:, 1] = 1.0
        self._pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._specified = torch.zeros_like(self._pending)
        self._specified_position = self.target_pos_w.clone()
        self._specified_angle = self.angle_rad.clone()
        self._specified_distance = self.distance_m.clone()
        self._state_cache = None
        self._state_counter = -1
        self._state_generation = 0
        self._cached_generation = -1
        task = env.cfg.task
        self._tracker = PushStateTracker(
            PushStateConfig(
                table_size_m=task.table_size, cube_size_m=task.cube_size,
                contact_threshold_n=task.contact_threshold_n,
                edge_margin_m=task.table_failure_margin_m,
                alignment_cos=math.cos(task.side_contact_angle_rad),
                approach_sigma_m=task.goal_reward_sigma_m,
                progress_sigma_fraction=task.progress_sigma_fraction,
                success_distance_m=task.success_distance_m,
                success_speed_m_s=task.success_speed_m_s,
                success_hold_time_s=task.success_hold_time_s,
                palm_height_offset_m=task.reset_position_offset_task[2],
            ), self.num_envs, self.device,
        )
        for name in ("goal_distance_m", "side_contact", "contact_seen", "push_seen", "settled_time_s", "success", "failure"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _ids(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

    @property
    def command(self):
        return torch.stack((torch.sin(self.angle_rad), torch.cos(self.angle_rad), self.distance_m), dim=-1)

    @property
    def tracker(self):
        return self._tracker

    def _directions(self, angle, env_ids):
        direction_base = torch.stack((angle.sin(), -angle.cos(), torch.zeros_like(angle)), dim=-1)
        _, base_quaternion = base_link_pose_w(self._env)
        return quaternion_rotate(base_quaternion[env_ids], direction_base)

    def _valid_path(self, positions, direction, distance, env_ids):
        task = self._env.cfg.task
        centre = positions.new_tensor(task.table_center[:2]) + self._env.scene.env_origins[env_ids, :2]
        half_extent = positions.new_tensor(task.table_size[:2]) / 2
        padding = math.sqrt(3.0) * task.cube_size / 2 + task.table_path_margin_m
        safe_extent = half_extent - padding
        goal = positions[:, :2] + distance[:, None] * direction[:, :2]
        # A straight segment between points inside a convex rectangle stays inside it.
        return ((positions[:, :2] - centre).abs() <= safe_extent + 1.e-7).all(-1) & (
            (goal - centre).abs() <= safe_extent + 1.e-7
        ).all(-1)

    def set_reset_specs(self, env_ids, cube_position, angle_rad, distance_m):
        """Set validated, one-shot reset fixtures; Cube positions are env-local.

        This is intended for reproducible smoke tests and still enforces the
        configured sampling ranges and conservative complete-path clearance.
        """

        index = self._ids(env_ids)
        if not len(index):
            return
        count = len(index)
        position = torch.as_tensor(cube_position, device=self.device, dtype=torch.float32).expand(count, 3).clone()
        angle = torch.as_tensor(angle_rad, device=self.device, dtype=torch.float32).expand(count).clone()
        distance = torch.as_tensor(distance_m, device=self.device, dtype=torch.float32).expand(count).clone()
        task = self._env.cfg.task
        finite = torch.isfinite(position).all() & torch.isfinite(angle).all() & torch.isfinite(distance).all()
        angle_to_side = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
        low, high = position.new_tensor(task.object_xy_range_low), position.new_tensor(task.object_xy_range_high)
        in_range = ((position[:, :2] >= low - 1.e-7) & (position[:, :2] <= high + 1.e-7)).all()
        in_range &= ((position[:, 2] - task.cube_center_height_m).abs() <= 1.e-6).all()
        in_range &= (angle_to_side.abs() <= task.command_angle_jitter_rad + 1.e-6).all()
        in_range &= ((distance >= task.command_distance_range[0] - 1.e-7) & (distance <= task.command_distance_range[1] + 1.e-7)).all()
        position += self._env.scene.env_origins[index]
        direction = self._directions(angle, index)
        if not bool(finite & in_range) or not bool(self._valid_path(position, direction, distance, index).all()):
            raise ValueError("Push reset specification must satisfy sampling bounds and complete Table path clearance")
        self._specified_position[index] = position
        self._specified_angle[index] = angle
        self._specified_distance[index] = distance
        self._specified[index] = True

    def sample_reset(self, env_ids):
        """Sample position, continuous angle, and length once before IK."""

        index = self._ids(env_ids)
        task = self._env.cfg.task
        fixed = index[self._specified[index]]
        self.sampled_cube_pos_w[fixed] = self._specified_position[fixed]
        self.angle_rad[fixed] = self._specified_angle[fixed]
        self.distance_m[fixed] = self._specified_distance[fixed]
        self.direction_w[fixed] = self._directions(self.angle_rad[fixed], fixed)
        pending = index[~self._specified[index]]
        low, high = self.target_pos_w.new_tensor(task.object_xy_range_low), self.target_pos_w.new_tensor(task.object_xy_range_high)
        for _ in range(task.command_sample_attempts):
            if not len(pending):
                break
            count = len(pending)
            position = torch.empty((count, 3), device=self.device)
            position[:, :2] = low + torch.rand((count, 2), device=self.device) * (high - low)
            position[:, 2] = task.cube_center_height_m
            position += self._env.scene.env_origins[pending]
            angle = (torch.rand(count, device=self.device) * 2 - 1) * task.command_angle_jitter_rad
            angle += torch.randint(0, 2, (count,), device=self.device) * math.pi
            distance = torch.empty(count, device=self.device).uniform_(*task.command_distance_range)
            direction = self._directions(angle, pending)
            valid = self._valid_path(position, direction, distance, pending)
            accepted = pending[valid]
            self.sampled_cube_pos_w[accepted] = position[valid]
            self.angle_rad[accepted] = angle[valid]
            self.distance_m[accepted] = distance[valid]
            self.direction_w[accepted] = direction[valid]
            pending = pending[~valid]
        if len(pending):
            raise RuntimeError(f"No valid complete Cube pushing path after {task.command_sample_attempts} attempts: env_ids={pending.tolist()}")
        self.target_pos_w[index] = self.sampled_cube_pos_w[index]
        self.goal_pos_w[index] = self.target_pos_w[index] + self.distance_m[index, None] * self.direction_w[index]
        self._pending[index] = True
        self._specified[index] = False
        return self.sampled_cube_pos_w[index]

    def _resample_command(self, env_ids):
        index = self._ids(env_ids)
        if not bool(self._pending[index].all()):
            raise RuntimeError("Push command reset requires the Cube placement event before IK and snapshot")
        super()._resample_command(index)
        self.goal_pos_w[index] = self.target_pos_w[index] + self.distance_m[index, None] * self.direction_w[index]
        self._pending[index] = False
        self._tracker.reset(index, self._snapshot())
        self._state_generation += 1

    def _snapshot(self):
        env = self._env
        cube = env.scene[self.cfg.object_name].data
        table = env.scene["table"].data
        palm_position, hand_quaternion = env.event_manager.get_term_cfg("safe_hand").func.palm_reference_pose_w()
        live = env.episode_length_buf > 0
        def forces(sensor_name):
            value = env.scene[sensor_name].data.force_matrix_w_history
            if value is None or value.shape[1] != env.cfg.decimation:
                raise RuntimeError(f"{sensor_name} must retain every physics substep's filtered contact history")
            return torch.where(live[:, None, None, None, None], value, torch.zeros_like(value))
        return PushSnapshot(
            initial_cube_pos_w=self.target_pos_w, goal_pos_w=self.goal_pos_w,
            direction_w=self.direction_w, distance_m=self.distance_m,
            cube_pos_w=cube.root_pos_w, cube_quat_w=cube.root_quat_w, cube_lin_vel_w=cube.root_lin_vel_w,
            palm_pos_w=palm_position, hand_quat_w=hand_quaternion,
            cube_palm_forces_w_history=forces("cube_palm_contacts"),
            cube_table_forces_w_history=forces("cube_table_contacts"),
            robot_table_failure=table_contact_mask(env),
            table_pos_w=table.root_pos_w, table_quat_w=table.root_quat_w,
            live=live,
        )

    def state(self):
        counter = self._env._sim_step_counter
        if self._state_counter != counter or self._cached_generation != self._state_generation:
            self._state_cache = self._tracker.update(self._snapshot(), physics_counter=counter, step_dt=self._env.step_dt)
            self._state_counter = counter
            self._cached_generation = self._state_generation
            self._record_metrics(self._state_cache)
        return self._state_cache

    def _update_metrics(self):
        # Manager computes metrics after auto-reset; reset rows must not advance
        # contact or stable-goal timers from the terminal episode's sensors.
        if not hasattr(self, "_tracker"):
            return
        self.state()

    def _record_metrics(self, state):
        self.metrics["goal_distance_m"][:] = state.cube_goal_distance_m
        self.metrics["side_contact"][:] = state.valid_push_contact.float()
        self.metrics["contact_seen"][:] = state.contact_seen.float()
        self.metrics["push_seen"][:] = state.push_seen.float()
        self.metrics["settled_time_s"][:] = state.settled_time_s
        self.metrics["success"][:] = state.success.float()
        self.metrics["failure"][:] = state.failure.float()

    def _set_debug_vis_impl(self, debug_vis):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and self._goal_marker is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._goal_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/HandManipulationPush/Goal",
                markers={"goal": sim_utils.SphereCfg(radius=.020, visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(.10, .90, .15), opacity=.65))},
            ))
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.95, .80, .05))
            self._direction_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/HandManipulationPush/Direction",
                markers={
                    "shaft": sim_utils.CylinderCfg(radius=.003, height=1.0, axis="Z", visual_material=material),
                    "tip": sim_utils.ConeCfg(radius=.008, height=.025, axis="Z", visual_material=material),
                },
            ))
        if self._goal_marker is not None:
            self._goal_marker.set_visibility(debug_vis)
            self._direction_marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if self._goal_marker is None or not hasattr(self, "goal_pos_w"):
            return
        self._goal_marker.visualize(translations=self.goal_pos_w)
        up = self.goal_pos_w.new_tensor((0., 0., .07))
        shaft_length = (self.distance_m - .025).clamp_min(.001)
        shaft = self.target_pos_w + .5 * shaft_length[:, None] * self.direction_w + up
        tip = self.goal_pos_w - .0125 * self.direction_w + up
        # Rotate procedural Z-axis shaft/cone onto the horizontal direction.
        quaternion = torch.zeros((self.num_envs, 4), device=self.device)
        quaternion[:, 0] = math.sqrt(.5)
        quaternion[:, 1] = -self.direction_w[:, 1] * math.sqrt(.5)
        quaternion[:, 2] = self.direction_w[:, 0] * math.sqrt(.5)
        scales = torch.ones((2 * self.num_envs, 3), device=self.device)
        scales[:self.num_envs, 2] = shaft_length
        self._direction_marker.visualize(
            translations=torch.cat((shaft, tip)), orientations=torch.cat((quaternion, quaternion)), scales=scales,
            marker_indices=torch.cat((torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
                                      torch.ones(self.num_envs, device=self.device, dtype=torch.long))),
        )


@configclass
class CubePushCommandCfg(CubeInitialPositionCommandCfg):
    class_type: type = CubePushCommand
    debug_vis: bool = True


__all__ = ["CubePushCommand", "CubePushCommandCfg"]
